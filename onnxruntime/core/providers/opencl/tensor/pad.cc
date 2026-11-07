// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pad.h"

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

    int64_t in_shape[5] = {1, 1, 1, 1, 1};
    int64_t out_shape[5] = {1, 1, 1, 1, 1};
    int64_t begin[5] = {0, 0, 0, 0, 0};
    std::vector<int64_t> out_dims(rank);
    for (int i = 0; i < rank; ++i) {
      in_shape[i] = xs[i];
      begin[i] = pads[i];
      out_shape[i] = xs[i] + pads[i] + pads[i + rank];
      out_dims[i] = out_shape[i];
    }
    Tensor* Y = context->Output(0, TensorShape(out_dims));
    LOGS_DEFAULT(ERROR) << "[OCL Pad] rank=" << rank << " last in=" << in_shape[rank - 1]
                       << " -> out=" << out_shape[rank - 1];

    const size_t nbytes = 5 * sizeof(int64_t);
    auto in_buf = exec_->GetScratchBufferTmp(nbytes);
    auto out_buf = exec_->GetScratchBufferTmp(nbytes);
    auto begin_buf = exec_->GetScratchBufferTmp(nbytes);
    exec_->WriteToCLBuffer(in_buf, in_shape, nbytes);
    exec_->WriteToCLBuffer(out_buf, out_shape, nbytes);
    exec_->WriteToCLBuffer(begin_buf, begin, nbytes);

    ORT_RETURN_IF_ERROR(
        KernelLauncher{GetKernel("Pad_Float")}
            .SetBuffers(*X, *Y)
            .SetBuffers(in_buf, out_buf, begin_buf)
            .SetArg<cl_int>(static_cast<cl_int>(rank))
            .SetArg<cl_float>(constant_value)
            .Launch(*exec_, {Y->Shape().Size(), 1, 1}));

    exec_->ReleaseCLBuffer(in_buf);
    exec_->ReleaseCLBuffer(out_buf);
    exec_->ReleaseCLBuffer(begin_buf);
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
