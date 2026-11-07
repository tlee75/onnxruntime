// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

__kernel void Exp_Float(
    __global const float* input,
    __global float* output) {
  int global_id = get_global_id(0);
  output[global_id] = exp(input[global_id]);
}
