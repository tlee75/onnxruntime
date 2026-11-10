// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "shape.h"

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

// The upstream branch reused the CPU Shape kernel with a CPU output
// (OutputMemoryType CPUInput). That CPU tensor is then consumed by GPU ops
// (Gather / Slice) in the same EP, which read it as a cl_mem -> the runtime
// fails with CL_INVALID_MEM_OBJECT. Emit the dims into a GPU buffer instead so
// the shape subgraph stays on the device.
class Shape : public OpenCLKernel {
 public:
  explicit Shape(const OpKernelInfo& info) : OpenCLKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const Tensor* X = ctx->Input<Tensor>(0);
    const auto dims = X->Shape().GetDims();
    Tensor* Y = ctx->Output(0, TensorShape({static_cast<int64_t>(dims.size())}));
    if (!dims.empty()) {
      exec_->WriteToCLBuffer(CL_BUFFER_FROM_TENSOR(*Y), dims.data(),
                             dims.size() * sizeof(int64_t));
    }
    return Status::OK();
  }
};

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    1, 12,
    kOpenCLExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Shape,
    kOnnxDomain,
    13, 14,
    kOpenCLExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

ONNX_OPERATOR_KERNEL_EX(
    Shape,
    kOnnxDomain,
    15,
    kOpenCLExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int64_t>()),
    Shape);

}  // namespace opencl
}  // namespace onnxruntime
