// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "convtranspose.h"

#include <algorithm>
#include <limits>
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
    if (exec_->UseFp16Kernels()) {
      LoadKernel("Gemm_Tiled_F16");
      // Half-*storage* twins for an fp16 graph (its tensors are half).
      LoadKernel("Gemm_Tiled_F16H");
      LoadKernel("ConvTranspose_Col2im_F16H");
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
