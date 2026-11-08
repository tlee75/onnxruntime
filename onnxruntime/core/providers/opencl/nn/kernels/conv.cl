// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Reuse the tiled GEMM (Gemm_Tiled) from the math kernels.
#include "math/kernels/gemm.cl"

// 1D convolution (cross-correlation), matching the ONNX Conv operator.
// X:   [N, C, W]        (row-major)
// Wt:  [M, C/group, K]
// B:   [M] (optional)
// Y:   [N, M, Wout], Wout = floor((W + pad_l + pad_r - dil*(K-1) - 1)/stride) + 1
//
// Naive fallback (one work-item per output, scalar inner loop). The fast path
// is im2col (Conv_Im2col) + Gemm_Tiled, used for group==1, batch==1.
__kernel void Conv1D_Float(
    __global const float* X,
    __global const float* Wt,
    __global const float* B,
    __global float* Y,
    const int N,
    const int C,
    const int W,
    const int M,
    const int K,
    const int Wout,
    const int group,
    const int stride,
    const int pad,
    const int dilation,
    const int has_bias) {
  int gid = get_global_id(0);
  if (gid >= N * M * Wout) {
    return;
  }
  int ow = gid % Wout;
  int t = gid / Wout;
  int m = t % M;
  int n = t / M;

  int Mg = M / group;
  int Cg = C / group;
  int g = m / Mg;
  int cbase = g * Cg;

  float acc = has_bias ? B[m] : 0.0f;
  for (int c = 0; c < Cg; ++c) {
    __global const float* xrow = X + (n * C + cbase + c) * W;
    __global const float* wrow = Wt + (m * Cg + c) * K;
    for (int k = 0; k < K; ++k) {
      int iw = ow * stride - pad + k * dilation;
      if (iw >= 0 && iw < W) {
        acc += xrow[iw] * wrow[k];
      }
    }
  }
  Y[gid] = acc;
}

// im2col for the fast path (group==1, one batch): unfold X [C, W] into
// col [C*K, Wout], where col[c*K + k, ow] = X[c, ow*stride - pad + k*dil]
// (zero outside the input). Then Y[M, Wout] = W[M, C*K] @ col.
//
// Materializing the unfold once is cheaper than gathering it inside the GEMM's
// B-tile load: the B tile is re-read M/GEMM_BM times, and a gather (with its
// index math and bounds checks) is slower per element than a coalesced read of
// a contiguous col — measured faster even though col costs an extra write.
//
// gws = (min(Wout, device cap), C*K). The `ow` axis GRID-STRIDES so the global
// work size never exceeds CL_DEVICE_MAX_WORK_ITEM_SIZES[0] — Mali tolerates a
// huge per-dimension size, Adreno does not (and schedules it pathologically).
__kernel void Conv_Im2col(
    const int C,
    const int W,
    const int K,
    const int Wout,
    const int stride,
    const int pad,
    const int dil,
    __global const float* X,
    __global float* col) {
  const int j = get_global_id(1);  // c*K + k
  if (j >= C * K) return;
  const int c = j / K;
  const int k = j - c * K;
  const int tap = k * dil;
  const int g0 = get_global_size(0);
  for (int ow = get_global_id(0); ow < Wout; ow += g0) {
    const int iw = ow * stride - pad + tap;
    col[j * Wout + ow] = (iw >= 0 && iw < W) ? X[c * W + iw] : 0.0f;
  }
}

// Add the per-output-channel bias to a Conv result: Y[m, ow] += B[m].
// gws = (min(Wout, device cap), M); the `ow` axis grid-strides (see above).
__kernel void Conv_AddBias(
    const int Wout,
    const int M,
    __global float* Y,
    __global const float* B) {
  const int m = get_global_id(1);
  if (m >= M) return;
  const int g0 = get_global_size(0);
  const float b = B[m];
  for (int ow = get_global_id(0); ow < Wout; ow += g0) {
    Y[m * Wout + ow] += b;
  }
}

// Fused conv GEMM: Y[M, Wout] = Wt[M, C*K] @ im2col(X[C, W]), with the im2col
// "B" operand GATHERED inline from X instead of materializing a col buffer.
//
// Templated on the tile so the host can pick a small-BM variant: the conv
// GEMMs are SHORT and WIDE (M is tiny -- down to 1 -- while Wout is huge), so
// a BM x BN accumulator tile wastes (BM - M) padded rows when M < BM.
#define DEFINE_GEMM_TILED_CONV(NAME, BM, BN, BK, TM, TN)                        \
  __kernel __attribute__((reqd_work_group_size(BN / TN, BM / TM, 1)))           \
  void NAME(                                                                    \
      const int M, const int Wout, const int C, const int W, const int K,       \
      const int stride, const int pad, const int dil,                           \
      __global const float* Wt, __global const float* X, __global float* Y) {   \
    const int lx = get_local_id(0);                                             \
    const int ly = get_local_id(1);                                             \
    const int tid = ly * get_local_size(0) + lx;                                \
    const int nthreads = get_local_size(0) * get_local_size(1);                 \
    __local float As[BM][BK];                                                   \
    __local float Bs[BK][BN];                                                   \
    const int row0 = get_group_id(1) * BM;                                      \
    const int Kdim = C * K;                                                     \
    const int numNtiles = (Wout + BN - 1) / BN;                                 \
    for (int nt = get_group_id(0); nt < numNtiles; nt += get_num_groups(0)) {   \
      const int col0 = nt * BN;                                                 \
      float acc[TM][TN];                                                        \
      for (int i = 0; i < TM; ++i)                                              \
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;                          \
      for (int k0 = 0; k0 < Kdim; k0 += BK) {                                   \
        for (int idx = tid; idx < BM * BK; idx += nthreads) {                   \
          const int i = idx / BK;                                               \
          const int kk = idx % BK;                                              \
          const int gr = row0 + i;                                              \
          const int gk = k0 + kk;                                               \
          As[i][kk] = (gr < M && gk < Kdim) ? Wt[gr * Kdim + gk] : 0.0f;        \
        }                                                                       \
        for (int idx = tid; idx < BK * BN; idx += nthreads) {                   \
          const int kk = idx / BN;                                              \
          const int j = idx % BN;                                               \
          const int gk = k0 + kk;                                               \
          const int ow = col0 + j;                                              \
          float v = 0.0f;                                                       \
          if (gk < Kdim && ow < Wout) {                                         \
            const int c = gk / K;                                               \
            const int tap = gk - c * K;                                         \
            const int iw = ow * stride - pad + tap * dil;                       \
            if (iw >= 0 && iw < W) v = X[c * W + iw];                           \
          }                                                                     \
          Bs[kk][j] = v;                                                        \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
        for (int kk = 0; kk < BK; ++kk) {                                       \
          float a[TM];                                                          \
          float b[TN];                                                          \
          for (int i = 0; i < TM; ++i) a[i] = As[ly * TM + i][kk];              \
          for (int j = 0; j < TN; ++j) b[j] = Bs[kk][lx * TN + j];              \
          for (int i = 0; i < TM; ++i)                                          \
            for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];              \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
      }                                                                         \
      for (int i = 0; i < TM; ++i) {                                            \
        const int gr = row0 + ly * TM + i;                                      \
        if (gr >= M) continue;                                                  \
        for (int j = 0; j < TN; ++j) {                                          \
          const int gc = col0 + lx * TN + j;                                    \
          if (gc < Wout) Y[gr * Wout + gc] = acc[i][j];                         \
        }                                                                       \
      }                                                                         \
      barrier(CLK_LOCAL_MEM_FENCE);                                             \
    }                                                                           \
  }

// The host picks by M (roughly: BM should be >= M but as small as possible).
DEFINE_GEMM_TILED_CONV(Gemm_Tiled_Conv, 64, 64, 32, 4, 4)
DEFINE_GEMM_TILED_CONV(Gemm_Tiled_Conv_BM32, 32, 64, 32, 4, 4)
DEFINE_GEMM_TILED_CONV(Gemm_Tiled_Conv_BM16, 16, 64, 32, 4, 4)
