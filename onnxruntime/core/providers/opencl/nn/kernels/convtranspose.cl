// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// 1D transposed convolution, matching the ONNX ConvTranspose operator.
// X:   [N, C, W]        (row-major)
// Wt:  [C, M/group, K]
// B:   [M] (optional)
// Y:   [N, M, Wout], Wout = (W-1)*stride - pad_l - pad_r + dil*(K-1) + 1 + output_padding
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
