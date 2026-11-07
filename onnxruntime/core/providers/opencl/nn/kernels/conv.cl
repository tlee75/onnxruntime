// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// 1D convolution (cross-correlation), matching the ONNX Conv operator.
// X:   [N, C, W]        (row-major)
// Wt:  [M, C/group, K]
// B:   [M] (optional)
// Y:   [N, M, Wout], Wout = floor((W + pad_l + pad_r - dil*(K-1) - 1)/stride) + 1
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
