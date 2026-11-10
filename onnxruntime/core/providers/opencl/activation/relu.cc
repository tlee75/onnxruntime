// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "relu.h"

#include <limits>

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME relu_kernel_src
#include "opencl_generated/activation/kernels/relu.cl.inc"
}  // namespace

class Relu : public OpenCLKernel {
 public:
  explicit Relu(const OpKernelInfo& info) : OpenCLKernel(info) {
    LoadProgram(relu_kernel_src, relu_kernel_src_len);
    LoadKernel("Relu_Float");
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    const int64_t input_size = X->Shape().Size();
    if (input_size == 0) {
      return Status::OK();
    }
    ORT_ENFORCE(input_size < std::numeric_limits<std::ptrdiff_t>::max());

    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("Relu_Float")}
            .SetBuffers(*X, *Y)
            .Launch(*exec_, {input_size, 1, 1}));

    return Status::OK();
  }
};

ONNX_OPENCL_OPERATOR_KERNEL(
    Relu,
    14,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Relu);

}  // namespace opencl
}  // namespace onnxruntime
