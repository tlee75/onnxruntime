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
    }
  }

  // The GEMM kernel to use: the fp16 variant when the device supports it AND the
  // session asked for fp16 (addOpenCL(useFp16, …)); otherwise the fp32 one.
  const char* GemmKernel() const {
    return (exec_->SupportsFp16() && exec_->UseFp16()) ? "Gemm_Tiled_F16" : "Gemm_Tiled";
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
            KernelLauncher{GetKernel(GemmKernel())}
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
        // General conv: fuse im2col into the GEMM. The conv GEMM is short and
        // wide, so pick a tile that fills the grid: a small BM when M is tiny
        // (wastes fewer padded rows), a narrow BN when Wout is small (more
        // N-tiles => more work-groups). The local size MUST match the variant.
        const size_t bm = (m_i <= 16) ? 16u : (m_i <= 32 ? 32u : 64u);
        const size_t bn = (bm == 64 && wout_i <= 128) ? 32u : 64u;
        // Pick the fp16 twin when the device supports fp16 and the session
        // asked for it; the fp16 kernels use half the local memory (better
        // occupancy) and half2 MAC.
        const bool f16 = exec_->SupportsFp16() && exec_->UseFp16();
        const char* kn = (bm == 16) ? (f16 ? "Gemm_Tiled_Conv_BM16_F16" : "Gemm_Tiled_Conv_BM16")
                                    : (bm == 32 ? (f16 ? "Gemm_Tiled_Conv_BM32_F16" : "Gemm_Tiled_Conv_BM32")
                                                : (bn == 32 ? (f16 ? "Gemm_Tiled_Conv_BN32_F16" : "Gemm_Tiled_Conv_BN32")
                                                            : (f16 ? "Gemm_Tiled_Conv_F16" : "Gemm_Tiled_Conv")));
        const size_t flx = bn / kGemmTN;
        const size_t fly = bm / kGemmTM;
        const size_t fgm = (static_cast<size_t>(m_i) + bm - 1) / bm;
        const size_t fgn = (static_cast<size_t>(wout_i) + bn - 1) / bn;
        const size_t fgn_used = std::min(fgn, std::max<size_t>(1, cap0 / flx));
        ORT_RETURN_IF_ERROR(
            KernelLauncher{GetKernel(kn)}
                .SetArg<cl_int>(m_i)
                .SetArg<cl_int>(wout_i)
                .SetArg<cl_int>(c_i)
                .SetArg<cl_int>(w_i)
                .SetArg<cl_int>(k_i)
                .SetArg<cl_int>(static_cast<cl_int>(stride))
                .SetArg<cl_int>(static_cast<cl_int>(pad_l))
                .SetArg<cl_int>(static_cast<cl_int>(dilation))
                .SetArg<cl_int>(has_bias)
                .SetBuffer(a)
                .SetBuffer(x)
                .SetBuffer(y)
                .SetBuffer(bias)
                .Launch(*exec_, NDRange(fgn_used * flx, fgm * fly), NDRange(flx, fly)));
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
