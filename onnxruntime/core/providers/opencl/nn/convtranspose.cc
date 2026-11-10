// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "convtranspose.h"

#include <vector>

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME convtranspose_kernel_src
#include "opencl_generated/nn/kernels/convtranspose.cl.inc"
}  // namespace

// 1D transposed convolution on the OpenCL EP (rank-3 input/weight only).
class ConvTranspose : public OpenCLKernel {
 public:
  explicit ConvTranspose(const OpKernelInfo& info) : OpenCLKernel(info) {
    info.GetAttrOrDefault<int64_t>("group", &group_, 1);
    strides_ = {1};
    pads_ = {0};
    dilations_ = {1};
    output_padding_ = {0};
    (void)info.GetAttrs("strides", strides_);
    (void)info.GetAttrs("pads", pads_);
    (void)info.GetAttrs("dilations", dilations_);
    (void)info.GetAttrs("output_padding", output_padding_);
    LoadProgram(convtranspose_kernel_src, convtranspose_kernel_src_len);
    LoadKernel("ConvTranspose1D_Float");
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

    cl_mem bias = (B != nullptr) ? CL_BUFFER_FROM_TENSOR(*B) : CL_BUFFER_FROM_TENSOR(*X);
    const int has_bias = (B != nullptr) ? 1 : 0;

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
            .SetArg<cl_int>(static_cast<cl_int>(has_bias))
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
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ConvTranspose);

}  // namespace opencl
}  // namespace onnxruntime
