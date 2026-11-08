// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// ELU: y = x if x > 0 else alpha * (exp(x) - 1).
__kernel void Elu_Float(
    __global const float* input,
    __global float* output,
    const float alpha) {
  int global_id = get_global_id(0);
  float v = input[global_id];
  output[global_id] = v > 0.0f ? v : alpha * (exp(v) - 1.0f);
}
