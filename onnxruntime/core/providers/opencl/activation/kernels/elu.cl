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

// fp16 twin: an fp16 graph's tensors are half, so the elementwise ops need
// half-buffer kernels (the activations are what dominate the fp16 tensor traffic).
// Guarded like the other fp16 kernels: `half` only exists under the fp16 prelude.
#ifdef USE_FP16
__kernel void Elu_Half(
    __global const half* input,
    __global half* output,
    const float alpha) {
  int global_id = get_global_id(0);
  const half v = input[global_id];
  output[global_id] = v > (half)0 ? v : (half)(alpha * (exp((float)v) - 1.0f));
}
#endif
