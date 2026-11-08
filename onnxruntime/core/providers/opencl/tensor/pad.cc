// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pad.h"

#include <cstdint>
#include <vector>

#include "core/providers/opencl/opencl_kernel.h"
#include "core/providers/opencl/opencl_utils.h"

namespace onnxruntime {
namespace opencl {

namespace {
#define CONTENT_NAME pad_kernel_src
#include "opencl_generated/tensor/kernels/pad.cl.inc"
}  // namespace

// Constant-mode padding (rank 1..5) on the OpenCL EP.
class Pad : public OpenCLKernel {
 public:
  explicit Pad(const OpKernelInfo& info) : OpenCLKernel(info) {
    LoadProgram(pad_kernel_src, pad_kernel_src_len);
    LoadKernel("Pad_Float");
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* X = context->Input<Tensor>(0);
    const Tensor* pads_t = context->Input<Tensor>(1);
    const Tensor* cv_t = context->Input<Tensor>(2);
    const auto& xs = X->Shape();
    const int rank = static_cast<int>(xs.NumDimensions());
    ORT_ENFORCE(rank >= 1 && rank <= 5, "OpenCL Pad supports rank 1..5");
    ORT_ENFORCE(pads_t != nullptr, "OpenCL Pad requires the pads input");
    const int64_t* pads = pads_t->Data<int64_t>();
    const float constant_value = (cv_t != nullptr) ? cv_t->Data<float>()[0] : 0.0f;

    int in_shape[5] = {1, 1, 1, 1, 1};
    int out_shape[5] = {1, 1, 1, 1, 1};
    int begin[5] = {0, 0, 0, 0, 0};
    std::vector<int64_t> out_dims(rank);
    for (int i = 0; i < rank; ++i) {
      in_shape[i] = static_cast<int>(xs[i]);
      begin[i] = static_cast<int>(pads[i]);
      out_shape[i] = static_cast<int>(xs[i] + pads[i] + pads[i + rank]);
      out_dims[i] = out_shape[i];
    }
    Tensor* Y = context->Output(0, TensorShape(out_dims));

    // Shape/pad metadata is constant for a fixed graph; upload it once (cached)
    // so it never inserts a blocking write into the pipeline. Keyed by this op
    // instance, so the cache stays bounded.
    const size_t nbytes = 5 * sizeof(int);
    const uint64_t base = reinterpret_cast<uint64_t>(this) * 8;
    cl_mem in_buf = exec_->GetOrUploadMetadata(base + 0, in_shape, nbytes);
    cl_mem out_buf = exec_->GetOrUploadMetadata(base + 1, out_shape, nbytes);
    cl_mem begin_buf = exec_->GetOrUploadMetadata(base + 2, begin, nbytes);

    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("Pad_Float")}
            .SetBuffers(*X, *Y)
            .SetBuffers(in_buf, out_buf, begin_buf)
            .SetArg<cl_int>(static_cast<cl_int>(rank))
            .SetArg<cl_float>(constant_value)
            .Launch(*exec_, {Y->Shape().Size(), 1, 1}));

    return Status::OK();
  }
};

ONNX_OPENCL_OPERATOR_KERNEL(
    Pad,
    13,
    KernelDefBuilder()
        .InputMemoryType(OrtMemTypeCPUInput, 1)  // pads
        .InputMemoryType(OrtMemTypeCPUInput, 2)  // constant_value
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pad);

}  // namespace opencl
}  // namespace onnxruntime
