// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Constant-mode N-D padding (rank <= 5). Pads are split as
// [begin_0..begin_{r-1}, end_0..end_{r-1}]; `begin` holds the begin pads.
__kernel void Pad_Float(
    __global const float* X,
    __global float* Y,
    __global const long* in_shape,
    __global const long* out_shape,
    __global const long* begin,
    const int rank,
    const float constant_value) {
  long gid = get_global_id(0);
  long out_idx[5];
  long rem = gid;
  for (int d = rank - 1; d >= 0; --d) {
    out_idx[d] = rem % out_shape[d];
    rem /= out_shape[d];
  }

  long in_index = 0;
  long stride = 1;
  int inside = 1;
  for (int d = rank - 1; d >= 0; --d) {
    long i = out_idx[d] - begin[d];
    if (i < 0 || i >= in_shape[d]) {
      inside = 0;
      break;
    }
    in_index += i * stride;
    stride *= in_shape[d];
  }
  Y[gid] = inside ? X[in_index] : constant_value;
}
