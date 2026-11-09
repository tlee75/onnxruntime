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
      // DEV-ONLY (ORT_OPENCL_EXP): see the variant selection in Compute().
      LoadKernel("Gemm_Tiled_Conv_H_TM8");
      LoadKernel("Gemm_Tiled_Conv_H_NG");
      LoadKernel("Gemm_Tiled_Conv_H_PB");
    }
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
        // DEV-ONLY A/B variants selected by ORT_OPENCL_EXP; both are half-storage and
        // use the same 64x64 tile/work-group, so only the kernel differs:
        //   conv_tm8      - TM=TN=8 (a 4x larger per-thread register block)
        //   conv_nogather - DIAGNOSTIC: a contiguous read that prices the strided
        //                   im2col gather (WRONG output; timing only)
        const char* exp_kernel = nullptr;
        int gtm = kGemmTM;
        int gtn = kGemmTN;
        if (h && exec_->Exp() == "conv_tm8") {
          exp_kernel = "Gemm_Tiled_Conv_H_TM8";
          gtm = gtn = 8;
        } else if (h && exec_->Exp() == "conv_nogather") {
          exp_kernel = "Gemm_Tiled_Conv_H_NG";
        } else if (h && exec_->Exp() == "conv_packb") {
          exp_kernel = "Gemm_Tiled_Conv_H_PB";
        }
        const ConvTile tile = exp_kernel ? ConvTile{64, 64}
                                         : ChooseConvTile(m_i, wout_i, cap0, f16, h, a, x, y, bias,
                                                          c_i, w_i, k_i, static_cast<int>(stride),
                                                          static_cast<int>(pad_l),
                                                          static_cast<int>(dilation), has_bias);
        const size_t flx = tile.bn / static_cast<size_t>(gtn);
        const size_t fly = tile.bm / static_cast<size_t>(gtm);
        const size_t fgm = (static_cast<size_t>(m_i) + tile.bm - 1) / tile.bm;
        const size_t fgn = (static_cast<size_t>(wout_i) + tile.bn - 1) / tile.bn;
        const size_t fgn_used = std::min(fgn, std::max<size_t>(1, cap0 / flx));
        KernelLauncher kl{GetKernel(exp_kernel ? exp_kernel : ConvTileKernel(tile, f16, h))};
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
