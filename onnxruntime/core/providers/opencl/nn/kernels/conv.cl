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
// gws = (Wout, C*K): the fast axis is `ow`, so the col stores are coalesced and
// (for stride 1) so are the X loads.
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
  const int ow = get_global_id(0);
  const int j = get_global_id(1);  // c*K + k
  if (ow >= Wout || j >= C * K) return;
  const int c = j / K;
  const int k = j % K;
  const int iw = ow * stride - pad + k * dil;
  col[j * Wout + ow] = (iw >= 0 && iw < W) ? X[c * W + iw] : 0.0f;
}

// Add the per-output-channel bias to a Conv result: Y[m, ow] += B[m].
// gws = (Wout, M).
__kernel void Conv_AddBias(
    const int Wout,
    const int M,
    __global float* Y,
    __global const float* B) {
  const int ow = get_global_id(0);
  const int m = get_global_id(1);
  if (ow >= Wout || m >= M) return;
  Y[m * Wout + ow] += B[m];
}
