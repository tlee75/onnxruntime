// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "elu.h"

#include <limits>

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME elu_kernel_src
#include "opencl_generated/activation/kernels/elu.cl.inc"
}  // namespace

// ELU. The vocoder export decomposes it into ~8 primitive ops per block
// (Neg/Relu/Neg/Relu/Exp/Sub/Relu/Sub); a single Elu op keeps it to one kernel.
class Elu : public OpenCLKernel {
 public:
  explicit Elu(const OpKernelInfo& info) : OpenCLKernel(info) {
    info.GetAttrOrDefault<float>("alpha", &alpha_, 1.0f);
    LoadProgram(elu_kernel_src, elu_kernel_src_len);
    LoadKernel("Elu_Float");
    if (exec_->SupportsFp16()) LoadKernel("Elu_Half");
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    Tensor* Y = context->Output(0, X->Shape());
    const int64_t input_size = X->Shape().Size();
    if (input_size == 0) {
      return Status::OK();
    }
    ORT_ENFORCE(input_size < std::numeric_limits<std::ptrdiff_t>::max());

    // Pick the kernel by the tensor's own dtype: an fp16 graph hands us half
    // buffers regardless of the session's use_fp16 (which only selects the
    // reduced-precision variants for fp32 tensors).
    const bool f16 = IsFp16Tensor(*X);
    const char* kn = f16 ? "Elu_Half" : "Elu_Float";
    // One line per kernel instance: the element type, the size, and what the
    // pointer test would have said. Getting an fp16 tensor but picking the fp32
    // kernel is invisible otherwise (it does not fail, it just computes garbage
    // from half-sized buffers).
    if (!dtype_logged_) {
      const auto* prim = X->DataType()->AsPrimitiveDataType();
      LOGS_DEFAULT(WARNING) << "[CL] Elu: elem_type=" << (prim ? prim->GetDataType() : -1)
                            << " fp16=" << f16
                            << " by_pointer=" << (X->DataType() == DataTypeImpl::GetTensorType<MLFloat16>())
                            << " bytes=" << X->SizeInBytes() << " elements=" << X->Shape().Size();
      dtype_logged_ = true;
    }

    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel(kn)}
            .SetBuffers(*X, *Y)
            .SetArg<cl_float>(alpha_)
            .Launch(*exec_, {input_size, 1, 1}));

    return Status::OK();
  }

 private:
  float alpha_ = 1.0f;
  // One-shot diagnostic (see Compute): mutable because Compute is const.
  mutable bool dtype_logged_ = false;
};

ONNX_OPENCL_OPERATOR_KERNEL(
    Elu,
    6,
    KernelDefBuilder()
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<MLFloat16>()}),
    Elu);

}  // namespace opencl
}  // namespace onnxruntime
