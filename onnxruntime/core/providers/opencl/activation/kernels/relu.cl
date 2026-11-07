// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

__kernel void Relu_Float(
    __global const float* input,
    __global float* output) {
  int global_id = get_global_id(0);
  float v = input[global_id];
  output[global_id] = v > 0.0f ? v : 0.0f;
}
