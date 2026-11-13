// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <mutex>
#include <vector>

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME conv_kernel_src
#include "opencl_generated/nn/kernels/conv.cl.inc"
}  // namespace

// Must match the tile sizes in gemm.cl (Gemm_Tiled).
static constexpr int kGemmBM = 64;
static constexpr int kGemmBN = 64;
static constexpr int kGemmTM = 4;
static constexpr int kGemmTN = 4;

// 1D convolution on the OpenCL EP. Only the cases the synth vocoder needs are
// supported (rank-3 input/weight, i.e. 1D); 2D is rejected by Compute.
//
// Fast path (group==1, batch==1): im2col the input into [C*K, Wout], then a
// tiled GEMM Y[M, Wout] = W[M, C*K] @ col, then add the bias. This is the
// vocoder's hot path. Everything else (group>1, batch>1, non-1D) falls back to
// the naive kernel.
class Conv : public OpenCLKernel {
 public:
  explicit Conv(const OpKernelInfo& info) : OpenCLKernel(info) {
    info.GetAttrOrDefault<int64_t>("group", &group_, 1);
    // NB: OpKernelInfo::GetAttrs APPENDS to the vector — do NOT pre-fill.
    strides_.clear();
    pads_.clear();
    dilations_.clear();
    (void)info.GetAttrs("strides", strides_);
    (void)info.GetAttrs("pads", pads_);
    (void)info.GetAttrs("dilations", dilations_);
    if (strides_.empty()) strides_ = {1};
    if (dilations_.empty()) dilations_ = {1};
    LoadProgram(conv_kernel_src, conv_kernel_src_len);
    LoadKernel("Conv1D_Float");
    LoadKernel("Conv_Im2col");
    LoadKernel("Gemm_Tiled");
    LoadKernel("Gemm_Tiled_Conv");
    LoadKernel("Gemm_Tiled_Conv_BM32");
    LoadKernel("Gemm_Tiled_Conv_BM16");
    LoadKernel("Gemm_Tiled_Conv_BN32");
    if (exec_->SupportsFp16()) {
      LoadKernel("Gemm_Tiled_F16");
      LoadKernel("Gemm_Tiled_Conv_F16");
      LoadKernel("Gemm_Tiled_Conv_BM32_F16");
      LoadKernel("Gemm_Tiled_Conv_BM16_F16");
      LoadKernel("Gemm_Tiled_Conv_BN32_F16");
      // Half-*storage* twins: for an fp16 graph the tensors themselves are half.
      LoadKernel("Gemm_Tiled_F16H");
      LoadKernel("Gemm_Tiled_Conv_H");
      LoadKernel("Gemm_Tiled_Conv_BM32_H");
      LoadKernel("Gemm_Tiled_Conv_BM16_H");
      LoadKernel("Gemm_Tiled_Conv_BN32_H");
      // The 1D Winograd F(4,3) path (see WinogradConv()).
      LoadKernel("WinoWeightT");
      LoadKernel("WinoInputT6");
      LoadKernel("WinoOutputT");
    }
  }

  // Release the cached Winograd scratch buffers (see the members below). Unlike
  // the session tensors -- owned by OpenCLBufferAllocator -- these are created
  // directly from the context below and are NOT registered with the EP's scratch
  // pool, so nothing else frees them. They are otherwise only released when
  // reallocated for a new shape, so every kernel destruction leaked them, and
  // with it the cl_context's device memory (~650 MB per GPU session). The base
  // ~OpenCLKernelHolder runs after this and releases the programs/kernels; the
  // context outlives both (the EP owns it).
  ~Conv() override {
    if (wino_u_) clReleaseMemObject(wino_u_);
    if (wino_v_) clReleaseMemObject(wino_v_);
    if (wino_p_) clReleaseMemObject(wino_p_);
  }

  // The GEMM kernel to use: the fp16 variant when the device supports it AND the
  // session asked for fp16 (addOpenCL(useFp16, …)); otherwise the fp32 one.
  const char* GemmKernel(bool h) const {
    if (h) return "Gemm_Tiled_F16H";
    return (exec_->SupportsFp16() && exec_->UseFp16()) ? "Gemm_Tiled_F16" : "Gemm_Tiled";
  }

  /// True when the tensor is half, i.e. an fp16 graph: its storage is half, so the
  /// *_F16H kernels (not the fp32-storage *_F16 ones) are the matching pair.
  static bool IsHalf(const Tensor& t) {
    return IsFp16Tensor(t);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    const Tensor* Wt = context->Input<Tensor>(1);
    const Tensor* B = context->Input<Tensor>(2);
    const auto& xs = X->Shape();
    const auto& ws = Wt->Shape();
    ORT_ENFORCE(xs.NumDimensions() == 3 && ws.NumDimensions() == 3,
                "OpenCL Conv supports 1D only (rank-3 input and weight)");

    const int64_t N = xs[0], C = xs[1], W = xs[2];
    const int64_t M = ws[0], K = ws[2];
    const int64_t stride = strides_.empty() ? 1 : strides_[0];
    const int64_t dilation = dilations_.empty() ? 1 : dilations_[0];
    const int64_t pad_l = pads_.size() > 0 ? pads_[0] : 0;
    const int64_t pad_r = pads_.size() > 1 ? pads_[1] : pad_l;
    const int64_t Wout = (W + pad_l + pad_r - dilation * (K - 1) - 1) / stride + 1;
    Tensor* Y = context->Output(0, TensorShape({N, M, Wout}));

    if (N * M * Wout == 0) {
      return Status::OK();
    }

    cl_mem bias = (B != nullptr) ? CL_BUFFER_FROM_TENSOR(*B) : CL_BUFFER_FROM_TENSOR(*X);
    const int has_bias = (B != nullptr) ? 1 : 0;

    if (group_ == 1 && N == 1) {
      // The conv kernels index in 32-bit; a dimension is never remotely near int,
      // but fail loudly rather than silently truncate through the casts below.
      const int64_t imax = std::numeric_limits<int>::max();
      ORT_ENFORCE(C <= imax && W <= imax && K <= imax && M <= imax && Wout <= imax,
                  "Conv: a dimension exceeds the OpenCL EP's 32-bit indexing");
      const int c_i = static_cast<int>(C);
      const int w_i = static_cast<int>(W);
      const int k_i = static_cast<int>(K);
      const int m_i = static_cast<int>(M);
      const int wout_i = static_cast<int>(Wout);
      const int kdim = c_i * k_i;
      // Per-dimension global work size cap (Adreno rejects huge per-dim sizes;
      // the kernels grid-stride over Wout).
      const size_t cap0 = exec_->MaxWorkItemSize(0);

      cl_mem a = CL_BUFFER_FROM_TENSOR(*Wt);  // [M, C*K] row-major
      cl_mem x = CL_BUFFER_FROM_TENSOR(*X);   // [C, W]
      cl_mem y = CL_BUFFER_FROM_TENSOR(*Y);

      // Y[M, Wout] = W[M, Kdim] @ im2col(X)
      const size_t lx = kGemmBN / kGemmTN;
      const size_t ly = kGemmBM / kGemmTM;
      const size_t gn = (static_cast<size_t>(wout_i) + kGemmBN - 1) / kGemmBN;
      const size_t gm = (static_cast<size_t>(m_i) + kGemmBM - 1) / kGemmBM;
      const size_t gn_used = std::min(gn, std::max<size_t>(1, cap0 / lx));

      if (k_i == 1 && stride == 1 && pad_l == 0) {
        // 1x1 conv: X is already [C, W] == [Kdim, Wout]; a plain GEMM.
        ORT_RETURN_IF_ERROR(
            KernelLauncher{GetKernel(GemmKernel(IsHalf(*X)))}
                .SetArg<cl_int>(m_i)
                .SetArg<cl_int>(wout_i)
                .SetArg<cl_int>(kdim)
                .SetArg<cl_int>(has_bias)
                .SetArg<cl_int>(0)  // trans_a: A is [M, Kdim] (the conv weight)
                .SetBuffer(a)
                .SetBuffer(x)
                .SetBuffer(y)
                .SetBuffer(bias)
                .SetArg<cl_int>(0)  // sa/sb/sc: a plain (single-plane) GEMM
                .SetArg<cl_int>(0)
                .SetArg<cl_int>(0)
                .Launch(*exec_, NDRange(gn_used * lx, gm * ly), NDRange(lx, ly)));
      } else {
        // General conv: fuse im2col into the GEMM.
        //
        // BM/BN set BOTH the register tile and the work-group shape (each
        // variant declares reqd_work_group_size), so choosing a tile means
        // choosing a kernel. The default below is the measured sweet spot for
        // this family of conv shapes; because that is a shape/device-tuned
        // choice rather than a derivation, `enableAutoTune` TIMES the
        // candidates on the first call for a shape and caches the winner --
        // other consumers do not have to trust the defaults.
        //
        // The fp16 twin is used when the device supports fp16 and the session
        // asked for it (half the local memory => better occupancy, half2 MAC).
        const bool f16 = exec_->SupportsFp16() && exec_->UseFp16();
        const bool h = IsHalf(*X);
        // 1D Winograd F(4,3) for any stride-1 k>=3 conv, decomposed into ceil(K/3)
        // 3-tap slices stacked along the reduction (K=3 is one slice, K=7 is three);
        // any pad works, slice s's window starts at 4j - pad + 3s. Half-storage
        // only, so the fp32 path is untouched -- the fp16 graph's Convs are where
        // this pays (docs/ORT_GPU_IMAGE2D_PROPOSAL.md section 8).
        const bool wino = h && k_i >= 3 && stride == 1 && dilation == 1 &&
                          (wout_i % 4) == 0 && m_i > 0 && c_i > 0;
        if (wino) {
          ORT_RETURN_IF_ERROR(WinogradConv(a, x, y, bias, m_i, wout_i, c_i, w_i, k_i,
                                           static_cast<int>(pad_l), has_bias));
          return Status::OK();
        }
        const ConvTile tile = ChooseConvTile(m_i, wout_i, cap0, f16, h, a, x, y, bias,
                                             c_i, w_i, k_i, static_cast<int>(stride),
                                             static_cast<int>(pad_l), static_cast<int>(dilation),
                                             has_bias);
        const size_t flx = tile.bn / static_cast<size_t>(kGemmTN);
        const size_t fly = tile.bm / static_cast<size_t>(kGemmTM);
        const size_t fgm = (static_cast<size_t>(m_i) + tile.bm - 1) / tile.bm;
        const size_t fgn = (static_cast<size_t>(wout_i) + tile.bn - 1) / tile.bn;
        const size_t fgn_used = std::min(fgn, std::max<size_t>(1, cap0 / flx));
        KernelLauncher kl{GetKernel(ConvTileKernel(tile, f16, h))};
        BindConvGemm(kl, a, x, y, bias, m_i, wout_i, c_i, w_i, k_i,
                     static_cast<int>(stride), static_cast<int>(pad_l),
                     static_cast<int>(dilation), has_bias);
        ORT_RETURN_IF_ERROR(
            kl.Launch(*exec_, NDRange(fgn_used * flx, fgm * fly), NDRange(flx, fly)));
      }
      return Status::OK();
    }

    // Naive fallback.
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("Conv1D_Float")}
            .SetBuffer(*X)
            .SetBuffer(*Wt)
            .SetBuffer(bias)
            .SetBuffer(*Y)
            .SetArg<cl_int>(static_cast<cl_int>(N))
            .SetArg<cl_int>(static_cast<cl_int>(C))
            .SetArg<cl_int>(static_cast<cl_int>(W))
            .SetArg<cl_int>(static_cast<cl_int>(M))
            .SetArg<cl_int>(static_cast<cl_int>(K))
            .SetArg<cl_int>(static_cast<cl_int>(Wout))
            .SetArg<cl_int>(static_cast<cl_int>(group_))
            .SetArg<cl_int>(static_cast<cl_int>(stride))
            .SetArg<cl_int>(static_cast<cl_int>(pad_l))
            .SetArg<cl_int>(static_cast<cl_int>(dilation))
            .SetArg<cl_int>(has_bias)
            .Launch(*exec_, {N * M * Wout, 1, 1}));

    return Status::OK();
  }


  // A fused-conv GEMM tile. The kernel is implied by (bm, bn, fp16).
  struct ConvTile {
    size_t bm, bn;
  };

  static const char* ConvTileKernel(const ConvTile& t, bool f16, bool h) {
    if (h) {
      if (t.bm == 16) return "Gemm_Tiled_Conv_BM16_H";
      if (t.bm == 32) return "Gemm_Tiled_Conv_BM32_H";
      if (t.bn == 32) return "Gemm_Tiled_Conv_BN32_H";
      return "Gemm_Tiled_Conv_H";
    }
    if (t.bm == 16) return f16 ? "Gemm_Tiled_Conv_BM16_F16" : "Gemm_Tiled_Conv_BM16";
    if (t.bm == 32) return f16 ? "Gemm_Tiled_Conv_BM32_F16" : "Gemm_Tiled_Conv_BM32";
    if (t.bn == 32) return f16 ? "Gemm_Tiled_Conv_BN32_F16" : "Gemm_Tiled_Conv_BN32";
    return f16 ? "Gemm_Tiled_Conv_F16" : "Gemm_Tiled_Conv";
  }

  static void BindConvGemm(KernelLauncher& kl, cl_mem a, cl_mem x, cl_mem y, cl_mem bias,
                           int m_i, int wout_i, int c_i, int w_i, int k_i,
                           int stride, int pad_l, int dilation, int has_bias) {
    kl.SetArg<cl_int>(m_i)
        .SetArg<cl_int>(wout_i)
        .SetArg<cl_int>(c_i)
        .SetArg<cl_int>(w_i)
        .SetArg<cl_int>(k_i)
        .SetArg<cl_int>(stride)
        .SetArg<cl_int>(pad_l)
        .SetArg<cl_int>(dilation)
        .SetArg<cl_int>(has_bias)
        .SetBuffer(a)
        .SetBuffer(x)
        .SetBuffer(y)
        .SetBuffer(bias);
  }

  // Pick the fused-conv GEMM tile: the default heuristic, or -- with
  // enableAutoTune -- whichever of the candidates is fastest for this shape
  // (timed once per shape and cached).
  ConvTile ChooseConvTile(int m_i, int wout_i, size_t cap0, bool f16, bool h,
                          cl_mem a, cl_mem x, cl_mem y, cl_mem bias,
                          int c_i, int w_i, int k_i, int stride, int pad_l, int dilation,
                          int has_bias) const {
    // Default (no autotune): BM is the smallest tile that still covers M, so no
    // accumulator row is padded away. BN is the full-width tile except when the
    // conv is narrow in BOTH M and Wout, where halving BN doubles the number of
    // N-tiles. That threshold is EMPIRICAL: Adreno and Mali are both sensitive to
    // the work-group count at these sizes (a 512x128 conv goes from 16 to 32
    // work-groups), and it was measured on Mali-G76 / Adreno 740.
    // enableAutoTune re-measures per device instead of trusting it.
    const ConvTile fallback = (m_i <= 16) ? ConvTile{16, 64}
                                          : (m_i <= 32 ? ConvTile{32, 64}
                                                       : ConvTile{64, wout_i <= 128 ? 32u : 64u});
    const std::array<int, 4> key{m_i, wout_i, c_i * k_i, f16 ? 1 : 0};
    {
      std::lock_guard<std::mutex> guard(tile_mu_);
      auto it = tile_choice_.find(key);
      if (it != tile_choice_.end()) {
        return ConvTile{it->second[0], it->second[1]};
      }
    }

    if (!exec_->TuneEnabledLevel()) {
      // Memoise + log once per shape, so a run records which tile each conv got.
      LOGS_DEFAULT(WARNING) << "[CL] conv gemm tile bm=" << fallback.bm << " bn=" << fallback.bn
                         << " M=" << m_i << " Wout=" << wout_i << " kdim=" << (c_i * k_i)
                         << (f16 ? " fp16" : " fp32");
      std::lock_guard<std::mutex> guard(tile_mu_);
      tile_choice_[key] = {fallback.bm, fallback.bn};
      return fallback;
    }

    static constexpr ConvTile kCandidates[] = {{64, 64}, {32, 64}, {16, 64}, {64, 32}};
    LOGS_DEFAULT(WARNING) << "[CL] conv gemm tuning: default bm=" << fallback.bm << " bn="
                       << fallback.bn << " M=" << m_i << " Wout=" << wout_i << " kdim="
                       << (c_i * k_i) << (f16 ? " fp16" : " fp32");

    ConvTile best = fallback;
    double best_us = std::numeric_limits<double>::max();
    for (const ConvTile& t : kCandidates) {
      const size_t flx = t.bn / kGemmTN;
      const size_t fly = t.bm / kGemmTM;
      const size_t fgm = (static_cast<size_t>(m_i) + t.bm - 1) / t.bm;
      const size_t fgn = (static_cast<size_t>(wout_i) + t.bn - 1) / t.bn;
      const size_t fgn_used = std::min(fgn, std::max<size_t>(1, cap0 / flx));
      KernelLauncher kl{GetKernel(ConvTileKernel(t, f16, h))};
      BindConvGemm(kl, a, x, y, bias, m_i, wout_i, c_i, w_i, k_i, stride, pad_l, dilation, has_bias);
      const double us = kl.TimeOnTuneQueue(*exec_, NDRange(fgn_used * flx, fgm * fly), NDRange(flx, fly));
      if (us < best_us) {
        best_us = us;
        best = t;
      }
    }
    LOGS_DEFAULT(WARNING) << "[CL] conv gemm tuned bm=" << best.bm << " bn=" << best.bn
                       << " (default was bm=" << fallback.bm << " bn=" << fallback.bn
                       << ") us=" << best_us << " M=" << m_i << " Wout=" << wout_i;
    {
      std::lock_guard<std::mutex> guard(tile_mu_);
      tile_choice_[key] = {best.bm, best.bn};
    }
    return best;
  }

 private:
  int64_t group_ = 1;
  std::vector<int64_t> strides_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> dilations_;

  // enableAutoTune: chosen fused-conv tile per (M, Wout, Kdim, fp16).
  mutable std::mutex tile_mu_;
  mutable std::map<std::array<int, 4>, std::array<size_t, 2>> tile_choice_;

  /// The 1D F(4,3) path for any stride-1 k>=3
  /// conv, as ceil(K/3) 3-tap slices stacked along the reduction (K=3 -> 1 slice,
  /// K=7 -> 3). The transformed weights U depend only on the shape, so they are
  /// built once and cached; V (input tiles) and P (transform-domain products) are
  /// cached per (M, C, T, NS) and rebuilt each call. All six transform points live
  /// in ONE U/V/P buffer (plane-stride apart) so the transform-domain GEMM is a
  /// SINGLE 3-D launch -- its z range is the transform point -- instead of six
  /// small ones. It reduces over NS*C instead of over C*K -- that is the win.
  Status WinogradConv(cl_mem wt, cl_mem x, cl_mem y, cl_mem bias,
                      int m_i, int wout_i, int c_i, int w_i, int k_i, int pad_l,
                      int has_bias) const {
    const int T = wout_i / 4;
    // ceil(K/3) 3-tap slices, stacked along the reduction: the transform-domain
    // GEMM then reduces over NS*C instead of over C*K.
    const int NS = (k_i + 2) / 3;
    const size_t nc = static_cast<size_t>(NS) * static_cast<size_t>(c_i);
    const size_t u_stride = static_cast<size_t>(m_i) * nc;
    const size_t v_stride = nc * static_cast<size_t>(T);
    const size_t p_stride = static_cast<size_t>(m_i) * static_cast<size_t>(T);
    const size_t u_sz = 6 * u_stride * sizeof(cl_half);
    const size_t v_sz = 6 * v_stride * sizeof(cl_half);
    const size_t p_sz = 6 * p_stride * sizeof(cl_half);
    std::lock_guard<std::mutex> guard(wino_mu_);
    if (wino_m_ != m_i || wino_c_ != c_i || wino_t_ != T || wino_ns_ != NS) {
      if (wino_u_) clReleaseMemObject(wino_u_);
      if (wino_v_) clReleaseMemObject(wino_v_);
      if (wino_p_) clReleaseMemObject(wino_p_);
      cl_int e1 = CL_SUCCESS, e2 = CL_SUCCESS, e3 = CL_SUCCESS;
      wino_u_ = clCreateBuffer(exec_->GetOpenCLContext(), CL_MEM_READ_WRITE, u_sz, nullptr, &e1);
      wino_v_ = clCreateBuffer(exec_->GetOpenCLContext(), CL_MEM_READ_WRITE, v_sz, nullptr, &e2);
      wino_p_ = clCreateBuffer(exec_->GetOpenCLContext(), CL_MEM_READ_WRITE, p_sz, nullptr, &e3);
      ORT_ENFORCE(e1 == CL_SUCCESS && e2 == CL_SUCCESS && e3 == CL_SUCCESS,
                  "WinogradConv: scratch allocation failed");
      wino_m_ = m_i;
      wino_c_ = c_i;
      wino_t_ = T;
      wino_ns_ = NS;
      for (int t = 0; t < 6; ++t) {
        ORT_RETURN_IF_ERROR(
            KernelLauncher{GetKernel("WinoWeightT")}
                .SetBuffer(wt)
                .SetArg<cl_int>(m_i)
                .SetArg<cl_int>(c_i)
                .SetArg<cl_int>(k_i)
                .SetArg<cl_int>(NS)
                .SetArg<cl_int>(t)
                .SetArg<cl_int>(static_cast<cl_int>(u_stride))
                .SetBuffer(wino_u_)
                .Launch(*exec_, NDRange(nc, static_cast<size_t>(m_i))));
      }
    }
    // All six transform points in ONE launch: each work-item reads its 6-tap
    // window once and emits all six planes (docs/ORT_GPU_IMAGE2D_PROPOSAL.md §8).
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("WinoInputT6")}
            .SetBuffer(x)
            .SetArg<cl_int>(c_i)
            .SetArg<cl_int>(w_i)
            .SetArg<cl_int>(NS)
            .SetArg<cl_int>(pad_l)
            .SetArg<cl_int>(T)
            .SetArg<cl_int>(static_cast<cl_int>(v_stride))
            .SetBuffer(wino_v_)
            .Launch(*exec_, NDRange(static_cast<size_t>(T), nc)));
    const size_t lx = 16;  // Gemm_Tiled_F16H work-group: GEMM_BN / GEMM_TN
    const size_t ly = 16;  //                             GEMM_BM / GEMM_TM
    const size_t gn = (static_cast<size_t>(T) + 63) / 64;
    const size_t gm = (static_cast<size_t>(m_i) + 63) / 64;
    const size_t cap0 = exec_->MaxWorkItemSize(0);
    const size_t gn_used = std::min(gn, std::max<size_t>(1, cap0 / lx));
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("Gemm_Tiled_F16H")}
            .SetArg<cl_int>(m_i)
            .SetArg<cl_int>(T)
            .SetArg<cl_int>(static_cast<cl_int>(nc))
            .SetArg<cl_int>(0)  // bias is applied by WinoOutputT
            .SetArg<cl_int>(0)  // trans_a: U is [M, NC]
            .SetBuffer(wino_u_)
            .SetBuffer(wino_v_)
            .SetBuffer(wino_p_)
            .SetBuffer(bias)
            .SetArg<cl_int>(static_cast<cl_int>(u_stride))
            .SetArg<cl_int>(static_cast<cl_int>(v_stride))
            .SetArg<cl_int>(static_cast<cl_int>(p_stride))
            .Launch(*exec_, NDRange(gn_used * lx, gm * ly, 6), NDRange(lx, ly, 1)));
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("WinoOutputT")}
            .SetBuffer(wino_p_)
            .SetArg<cl_int>(static_cast<cl_int>(p_stride))
            .SetArg<cl_int>(m_i)
            .SetArg<cl_int>(T)
            .SetArg<cl_int>(has_bias)
            .SetBuffer(bias)
            .SetBuffer(y)
            .SetArg<cl_int>(wout_i)
            .SetArg<cl_int>(1)  // os: a plain conv writes every element
            .SetArg<cl_int>(0)  // r
            .Launch(*exec_, NDRange(static_cast<size_t>(T), static_cast<size_t>(m_i))));
    return Status::OK();
  }

  // Cached transform-domain tensors, six planes back to back per buffer, rebuilt
  // when the shape changes and kept for the session (see WinogradConv()).
  mutable std::mutex wino_mu_;
  mutable cl_mem wino_u_ = nullptr;
  mutable cl_mem wino_v_ = nullptr;
  mutable cl_mem wino_p_ = nullptr;
  mutable int wino_m_ = 0;
  mutable int wino_c_ = 0;
  mutable int wino_t_ = 0;
  mutable int wino_ns_ = 0;
};

ONNX_OPENCL_OPERATOR_KERNEL(
    Conv,
    11,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()}),
    Conv);

}  // namespace opencl
}  // namespace onnxruntime
