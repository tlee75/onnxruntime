// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv.h"

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
    LoadKernel("Conv_AddBias");
    LoadKernel("Gemm_Tiled");
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
      const int c_i = static_cast<int>(C);
      const int w_i = static_cast<int>(W);
      const int k_i = static_cast<int>(K);
      const int m_i = static_cast<int>(M);
      const int wout_i = static_cast<int>(Wout);
      const int kdim = c_i * k_i;

      cl_mem a = CL_BUFFER_FROM_TENSOR(*Wt);  // [M, C*K] row-major
      cl_mem y = CL_BUFFER_FROM_TENSOR(*Y);
      cl_mem b = nullptr;
      bool own_b = false;
      if (k_i == 1 && stride == 1 && pad_l == 0) {
        // 1x1 conv: X is already [C, W] == [Kdim, Wout]; no unfold needed.
        b = CL_BUFFER_FROM_TENSOR(*X);
      } else {
        b = exec_->GetScratchBufferTmp(static_cast<size_t>(kdim) * wout_i * sizeof(float));
        own_b = true;
        ORT_RETURN_IF_ERROR(
            KernelLauncher{GetKernel("Conv_Im2col")}
                .SetArg<cl_int>(c_i)
                .SetArg<cl_int>(w_i)
                .SetArg<cl_int>(k_i)
                .SetArg<cl_int>(wout_i)
                .SetArg<cl_int>(static_cast<cl_int>(stride))
                .SetArg<cl_int>(static_cast<cl_int>(pad_l))
                .SetArg<cl_int>(static_cast<cl_int>(dilation))
                .SetBuffer(CL_BUFFER_FROM_TENSOR(*X))
                .SetBuffer(b)
                .Launch(*exec_, NDRange(static_cast<size_t>(wout_i), static_cast<size_t>(kdim))));
      }

      // Y[M, Wout] = W[M, Kdim] @ col[Kdim, Wout]
      const size_t lx = kGemmBN / kGemmTN;
      const size_t ly = kGemmBM / kGemmTM;
      const size_t gn = (static_cast<size_t>(wout_i) + kGemmBN - 1) / kGemmBN;
      const size_t gm = (static_cast<size_t>(m_i) + kGemmBM - 1) / kGemmBM;
      ORT_RETURN_IF_ERROR(
          KernelLauncher{GetKernel("Gemm_Tiled")}
              .SetArg<cl_int>(m_i)
              .SetArg<cl_int>(wout_i)
              .SetArg<cl_int>(kdim)
              .SetBuffer(a)
              .SetBuffer(b)
              .SetBuffer(y)
              .Launch(*exec_, NDRange(gn * lx, gm * ly), NDRange(lx, ly)));

      if (own_b) {
        exec_->ReleaseCLBuffer(b);
      }

      if (has_bias) {
        ORT_RETURN_IF_ERROR(
            KernelLauncher{GetKernel("Conv_AddBias")}
                .SetArg<cl_int>(wout_i)
                .SetArg<cl_int>(m_i)
                .SetBuffer(y)
                .SetBuffer(bias)
                .Launch(*exec_, NDRange(static_cast<size_t>(wout_i), static_cast<size_t>(m_i))));
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

 private:
  int64_t group_ = 1;
  std::vector<int64_t> strides_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> dilations_;
};

ONNX_OPENCL_OPERATOR_KERNEL(
    Conv,
    11,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv);

}  // namespace opencl
}  // namespace onnxruntime
