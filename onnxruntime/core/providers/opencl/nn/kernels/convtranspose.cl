// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Reuse the tiled GEMM (Gemm_Tiled) from the math kernels.
#include "math/kernels/gemm.cl"

// 1D transposed convolution, matching the ONNX ConvTranspose operator.
// X:   [N, C, W]        (row-major)
// Wt:  [C, M/group, K]
// B:   [M] (optional)
// Y:   [N, M, Wout], Wout = (W-1)*stride - pad_l - pad_r + dil*(K-1) + 1 + output_padding
//
// Naive fallback (one work-item per output). The fast path is
// transpose-weight + Gemm_Tiled + col2im, used for group==1, batch==1.
__kernel void ConvTranspose1D_Float(
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
  int mlocal = m - g * Mg;

  float acc = has_bias ? B[m] : 0.0f;
  for (int c = 0; c < Cg; ++c) {
    int gc = cbase + c;
    __global const float* xrow = X + (n * C + gc) * W;
    __global const float* wrow = Wt + (gc * Mg + mlocal) * K;
    for (int k = 0; k < K; ++k) {
      int num = ow + pad - k * dilation;
      if (num % stride == 0) {
        int iw = num / stride;
        if (iw >= 0 && iw < W) {
          acc += xrow[iw] * wrow[k];
        }
      }
    }
  }
  Y[gid] = acc;
}

// Fast path (group==1, one batch):
//   1. transpose the weight [C, M, K] -> WtT [M*K, C]
//   2. P[M*K, W] = WtT[M*K, C] @ X[C, W]                 (Gemm_Tiled)
//   3. Y[m, ow] = B[m] + sum_k P[m*K + k, (ow + pad - k*dil)/stride]
//      (only the k with (ow+pad-k*dil) divisible by stride and in range).
// The col2im step is a gather (no atomics), so overlapping taps are handled.
//
// gws = (C, M*K): the writes to WtT are coalesced.
__kernel void ConvTranspose_TransposeWeight(
    const int C,
    const int M,
    const int K,
    __global const float* Wt,
    __global float* WtT) {
  const int c = get_global_id(0);
  const int mk = get_global_id(1);  // m*K + k
  if (c >= C || mk >= M * K) return;
  WtT[mk * C + c] = Wt[c * (M * K) + mk];
}

// col2im gather: gws = (Wout, M).
__kernel void ConvTranspose_Col2im(
    const int M,
    const int K,
    const int W,
    const int Wout,
    const int stride,
    const int pad,
    const int dil,
    const int has_bias,
    __global const float* P,
    __global const float* B,
    __global float* Y) {
  const int ow = get_global_id(0);
  const int m = get_global_id(1);
  if (ow >= Wout || m >= M) return;
  float acc = has_bias ? B[m] : 0.0f;
  for (int k = 0; k < K; ++k) {
    const int num = ow + pad - k * dil;
    if (num >= 0 && (num % stride) == 0) {
      const int iw = num / stride;
      if (iw < W) acc += P[(m * K + k) * W + iw];
    }
  }
  Y[m * Wout + ow] = acc;
}
