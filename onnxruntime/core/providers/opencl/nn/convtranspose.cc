// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "convtranspose.h"

#include <algorithm>
#include <array>
#include <limits>
#include <mutex>
#include <vector>

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME convtranspose_kernel_src
#include "opencl_generated/nn/kernels/convtranspose.cl.inc"
}  // namespace

// Must match the tile sizes in gemm.cl (Gemm_Tiled).
static constexpr int kGemmBM = 64;
static constexpr int kGemmBN = 64;
static constexpr int kGemmTM = 4;
static constexpr int kGemmTN = 4;

// 1D transposed convolution on the OpenCL EP (rank-3 input/weight only).
//
// Fast path (group==1, batch==1): transpose the weight to [M*K, C], one tiled
// GEMM P[M*K, W] = WtT[M*K, C] @ X[C, W], then a col2im gather into Y. This is
// the vocoder's hot path. Everything else falls back to the naive kernel.
class ConvTranspose : public OpenCLKernel {
 public:
  explicit ConvTranspose(const OpKernelInfo& info) : OpenCLKernel(info) {
    info.GetAttrOrDefault<int64_t>("group", &group_, 1);
    // NB: OpKernelInfo::GetAttrs APPENDS to the vector — do NOT pre-fill.
    strides_.clear();
    pads_.clear();
    dilations_.clear();
    output_padding_.clear();
    (void)info.GetAttrs("strides", strides_);
    (void)info.GetAttrs("pads", pads_);
    (void)info.GetAttrs("dilations", dilations_);
    (void)info.GetAttrs("output_padding", output_padding_);
    if (strides_.empty()) strides_ = {1};
    if (dilations_.empty()) dilations_ = {1};
    LoadProgram(convtranspose_kernel_src, convtranspose_kernel_src_len);
    LoadKernel("ConvTranspose1D_Float");
    LoadKernel("ConvTranspose_Col2im");
    LoadKernel("Gemm_Tiled");
    if (exec_->SupportsFp16()) {
      LoadKernel("Gemm_Tiled_F16");
      // Half-*storage* twins for an fp16 graph (its tensors are half).
      LoadKernel("Gemm_Tiled_F16H");
      LoadKernel("ConvTranspose_Col2im_F16H");
      // DEV-ONLY (ORT_OPENCL_EXP=winoct): the phase-Winograd path in Compute().
      LoadKernel("WinoCtGather");
      LoadKernel("WinoWeightT");
      LoadKernel("WinoInputT");
      LoadKernel("WinoOutputT");
    }
  }

  // See Conv::GemmKernel: fp16 only when the device supports it AND the session
  // asked for fp16.
  const char* GemmKernel(bool h) const {
    if (h) return "Gemm_Tiled_F16H";
    return (exec_->SupportsFp16() && exec_->UseFp16()) ? "Gemm_Tiled_F16" : "Gemm_Tiled";
  }

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
                "OpenCL ConvTranspose supports 1D only (rank-3 input and weight)");

    const int64_t N = xs[0], C = xs[1], W = xs[2];
    const int64_t M = ws[1] * group_;  // weight is [C, M/group, K]
    const int64_t K = ws[2];
    const int64_t stride = strides_.empty() ? 1 : strides_[0];
    const int64_t dilation = dilations_.empty() ? 1 : dilations_[0];
    const int64_t pad_l = pads_.size() > 0 ? pads_[0] : 0;
    const int64_t pad_r = pads_.size() > 1 ? pads_[1] : pad_l;
    const int64_t out_pad = output_padding_.empty() ? 0 : output_padding_[0];
    const int64_t Wout =
        (W - 1) * stride - pad_l - pad_r + dilation * (K - 1) + 1 + out_pad;
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
                  "ConvTranspose: a dimension exceeds the OpenCL EP's 32-bit indexing");
      const int c_i = static_cast<int>(C);
      const int w_i = static_cast<int>(W);
      const int k_i = static_cast<int>(K);
      const int m_i = static_cast<int>(M);
      const int wout_i = static_cast<int>(Wout);
      const int mk = m_i * k_i;  // rows of WtT / P
      const size_t cap0 = exec_->MaxWorkItemSize(0);  // per-dim global size cap

      cl_mem wt = CL_BUFFER_FROM_TENSOR(*Wt);
      cl_mem x = CL_BUFFER_FROM_TENSOR(*X);
      cl_mem y = CL_BUFFER_FROM_TENSOR(*Y);

      // DEV-ONLY (ORT_OPENCL_EXP=winoct): a stride-s transposed conv splits into s
      // PHASE convs of ~K/s taps; here K == 2s, so each phase is exactly 2 taps --
      // and a 2-tap filter is a 3-tap one with a zero third, so each phase runs the
      // verified F(4,3) pipeline (NS=1) with its own pad and writes every os-th
      // output element directly (no col2im). 6 multiplies / 4 outputs vs 8 for the
      // direct 2-tap, i.e. 1.33x.
      if (IsHalf(*X) && exec_->Exp().find("winoct") != std::string::npos &&
          stride > 1 && dilation == 1 && K == 2 * stride) {
        ORT_RETURN_IF_ERROR(WinogradCtPhases(wt, x, y, bias, has_bias, m_i, wout_i,
                                             c_i, w_i, k_i, static_cast<int>(stride),
                                             static_cast<int>(pad_l)));
        return Status::OK();
      }

      cl_mem p = exec_->GetScratchBufferTmp(static_cast<size_t>(mk) * w_i * sizeof(float));

      // P[M*K, W] = Wt[C, M*K]^T @ X[C, W]. The GEMM reads the weight straight
      // from its [C, M*K] layout (trans_a=1) instead of materializing a
      // transposed copy first -- the transpose was a whole extra kernel plus a
      // M*K*C read+write every run, for a matrix that never changes.
      const size_t lx = kGemmBN / kGemmTN;
      const size_t ly = kGemmBM / kGemmTM;
      const size_t gn = (static_cast<size_t>(w_i) + kGemmBN - 1) / kGemmBN;
      const size_t gm = (static_cast<size_t>(mk) + kGemmBM - 1) / kGemmBM;
      const size_t gn_used = std::min(gn, std::max<size_t>(1, cap0 / lx));
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel(GemmKernel(IsHalf(*X)))}
              .SetArg<cl_int>(mk)
              .SetArg<cl_int>(w_i)
              .SetArg<cl_int>(c_i)
              .SetArg<cl_int>(0)  // bias applied by Col2im below
              .SetArg<cl_int>(1)  // trans_a: A is the [C, M*K] weight
              .SetBuffer(wt)
              .SetBuffer(x)
              .SetBuffer(p)
              .SetBuffer(bias)
              .Launch(*exec_, NDRange(gn_used * lx, gm * ly), NDRange(lx, ly)));

      // Y[m, ow] = B[m] + sum_k P[m*K + k, (ow + pad - k*dil)/stride]
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel(IsHalf(*X) ? "ConvTranspose_Col2im_F16H" : "ConvTranspose_Col2im")}
              .SetArg<cl_int>(m_i)
              .SetArg<cl_int>(k_i)
              .SetArg<cl_int>(w_i)
              .SetArg<cl_int>(wout_i)
              .SetArg<cl_int>(static_cast<cl_int>(stride))
              .SetArg<cl_int>(static_cast<cl_int>(pad_l))
              .SetArg<cl_int>(static_cast<cl_int>(dilation))
              .SetArg<cl_int>(has_bias)
              .SetBuffer(p)
              .SetBuffer(bias)
              .SetBuffer(y)
              .Launch(*exec_, NDRange(std::min<size_t>(static_cast<size_t>(wout_i), cap0),
                                      static_cast<size_t>(m_i))));

      exec_->ReleaseCLBuffer(p);
      return Status::OK();
    }

    // Naive fallback.
    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("ConvTranspose1D_Float")}
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

 private:
  int64_t group_ = 1;
  std::vector<int64_t> strides_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> dilations_;
  std::vector<int64_t> output_padding_;

  // DEV-ONLY (ORT_OPENCL_EXP=winoct): per-phase transform-domain tensors, cached
  // per shape. Never released (process-lifetime; dev only).
  mutable std::mutex ct_mu_;
  mutable std::vector<std::array<cl_mem, 6>> ct_u_, ct_v_, ct_p_;
  mutable std::vector<cl_mem> ct_wp_;
  mutable std::vector<long> ct_key_;

  /// DEV-ONLY (ORT_OPENCL_EXP=winoct): the ConvTranspose phase path (see Compute).
  Status WinogradCtPhases(cl_mem wt, cl_mem x, cl_mem y, cl_mem bias, int has_bias,
                          int m_i, int wout_i, int c_i, int w_i, int k_i,
                          int os, int pad_l) const {
    if (static_cast<int>(ct_key_.size()) < os) {
      ct_u_.resize(os);
      ct_v_.resize(os);
      ct_p_.resize(os);
      ct_wp_.resize(os, nullptr);
      ct_key_.resize(os, 0);
    }
    std::lock_guard<std::mutex> guard(ct_mu_);
    for (int r = 0; r < os; ++r) {
      const int k0 = (r + pad_l) % os;
      const int a0 = (r + pad_l - k0) / os;
      const int pad3 = 1 - a0;  // the phase, as a 3-tap correlation's pad
      const int Q = (wout_i - r + os - 1) / os;
      if (Q <= 0) continue;
      const int T = (Q + 3) / 4;
      const size_t mc = static_cast<size_t>(m_i) * static_cast<size_t>(c_i);
      const size_t ct = static_cast<size_t>(c_i) * static_cast<size_t>(T);
      const size_t mt = static_cast<size_t>(m_i) * static_cast<size_t>(T);
      const long key = static_cast<long>(m_i) * 1000000007L + static_cast<long>(c_i) * 100003L + T;
      if (ct_key_[r] != key) {
        for (int t = 0; t < 6; ++t) {
          if (ct_u_[r][t]) clReleaseMemObject(ct_u_[r][t]);
          if (ct_v_[r][t]) clReleaseMemObject(ct_v_[r][t]);
          if (ct_p_[r][t]) clReleaseMemObject(ct_p_[r][t]);
          cl_int e = CL_SUCCESS;
          ct_u_[r][t] = clCreateBuffer(exec_->GetOpenCLContext(), CL_MEM_READ_WRITE, mc * 3 * sizeof(cl_half), nullptr, &e);
          ct_v_[r][t] = clCreateBuffer(exec_->GetOpenCLContext(), CL_MEM_READ_WRITE, ct * sizeof(cl_half), nullptr, &e);
          ct_p_[r][t] = clCreateBuffer(exec_->GetOpenCLContext(), CL_MEM_READ_WRITE, mt * sizeof(cl_half), nullptr, &e);
          ORT_ENFORCE(e == CL_SUCCESS, "WinogradCtPhases: scratch allocation failed");
        }
        if (ct_wp_[r]) clReleaseMemObject(ct_wp_[r]);
        cl_int e = CL_SUCCESS;
        ct_wp_[r] = clCreateBuffer(exec_->GetOpenCLContext(), CL_MEM_READ_WRITE, mc * 3 * sizeof(cl_half), nullptr, &e);
        ORT_ENFORCE(e == CL_SUCCESS, "WinogradCtPhases: scratch allocation failed");
        // This phase's two taps -> Wp[M][C][3]; U depends only on Wp, so both are
        // built once per shape.
        ORT_RETURN_IF_ERROR(
            KernelLauncher{GetKernel("WinoCtGather")}
                .SetBuffer(wt)
                .SetArg<cl_int>(m_i)
                .SetArg<cl_int>(c_i)
                .SetArg<cl_int>(k_i)
                .SetArg<cl_int>(os)
                .SetArg<cl_int>(k0)
                .SetBuffer(ct_wp_[r])
                .Launch(*exec_, NDRange(static_cast<size_t>(c_i), static_cast<size_t>(m_i))));
        for (int t = 0; t < 6; ++t) {
          ORT_RETURN_IF_ERROR(
              KernelLauncher{GetKernel("WinoWeightT")}
                  .SetBuffer(ct_wp_[r])
                  .SetArg<cl_int>(m_i)
                  .SetArg<cl_int>(c_i)
                  .SetArg<cl_int>(3)  // K
                  .SetArg<cl_int>(1)  // NS
                  .SetArg<cl_int>(t)
                  .SetBuffer(ct_u_[r][t])
                  .Launch(*exec_, NDRange(static_cast<size_t>(c_i), static_cast<size_t>(m_i))));
        }
        ct_key_[r] = key;
      }
      for (int t = 0; t < 6; ++t) {
        ORT_RETURN_IF_ERROR(
            KernelLauncher{GetKernel("WinoInputT")}
                .SetBuffer(x)
                .SetArg<cl_int>(c_i)
                .SetArg<cl_int>(w_i)
                .SetArg<cl_int>(1)  // NS
                .SetArg<cl_int>(pad3)
                .SetArg<cl_int>(T)
                .SetArg<cl_int>(t)
                .SetBuffer(ct_v_[r][t])
                .Launch(*exec_, NDRange(static_cast<size_t>(T), static_cast<size_t>(c_i))));
      }
      const size_t lx = 16;  // Gemm_Tiled_F16H work-group: GEMM_BN / GEMM_TN
      const size_t ly = 16;  //                             GEMM_BM / GEMM_TM
      const size_t gn = (static_cast<size_t>(T) + 63) / 64;
      const size_t gm = (static_cast<size_t>(m_i) + 63) / 64;
      const size_t cap0 = exec_->MaxWorkItemSize(0);
      const size_t gn_used = std::min(gn, std::max<size_t>(1, cap0 / lx));
      for (int t = 0; t < 6; ++t) {
        ORT_RETURN_IF_ERROR(
            KernelLauncher{GetKernel("Gemm_Tiled_F16H")}
                .SetArg<cl_int>(m_i)
                .SetArg<cl_int>(T)
                .SetArg<cl_int>(c_i)
                .SetArg<cl_int>(0)  // bias is applied by WinoOutputT
                .SetArg<cl_int>(0)  // trans_a: U is [M, C]
                .SetBuffer(ct_u_[r][t])
                .SetBuffer(ct_v_[r][t])
                .SetBuffer(ct_p_[r][t])
                .SetBuffer(bias)
                .Launch(*exec_, NDRange(gn_used * lx, gm * ly), NDRange(lx, ly)));
      }
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel("WinoOutputT")}
              .SetBuffer(ct_p_[r][0])
              .SetBuffer(ct_p_[r][1])
              .SetBuffer(ct_p_[r][2])
              .SetBuffer(ct_p_[r][3])
              .SetBuffer(ct_p_[r][4])
              .SetBuffer(ct_p_[r][5])
              .SetArg<cl_int>(m_i)
              .SetArg<cl_int>(T)
              .SetArg<cl_int>(has_bias)
              .SetBuffer(bias)
              .SetBuffer(y)
              .SetArg<cl_int>(wout_i)
              .SetArg<cl_int>(os)
              .SetArg<cl_int>(r)
              .Launch(*exec_, NDRange(static_cast<size_t>(T), static_cast<size_t>(m_i))));
    }
    return Status::OK();
  }
};

ONNX_OPENCL_OPERATOR_KERNEL(
    ConvTranspose,
    11,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()}),
    ConvTranspose);

}  // namespace opencl
}  // namespace onnxruntime
