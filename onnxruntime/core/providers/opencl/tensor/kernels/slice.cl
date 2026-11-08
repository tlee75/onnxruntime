// Shapes/strides are int32 element counts and all index math is 32-bit: the
// per-dim values fit int for any tensor the OpenCL EP accepts, and 64-bit
// integer division is pathologically slow on some mobile GPUs (e.g. Adreno).
__kernel void Slice(__global uchar* input,
                    __global uchar* output,
                    int input_dimensions,
                    int element_size,
                    __global int* input_starts,
                    __global int* input_steps,
                    __global int* input_strides,
                    __global int* output_strides,
                    int output_dimensions,
                    int max_input_size) {
  int global_id = get_global_id(0);

  int output_idx = global_id;
  int input_offset = 0;

  for (int i = 0; i < output_dimensions; ++i) {
    int coord = output_idx / output_strides[i];
    output_idx %= output_strides[i];

    int input_coord = input_starts[i] + coord * input_steps[i];
    input_offset += input_coord * input_strides[i];
    if (input_offset * element_size >= max_input_size) {
      return;
    }
  }

  for (int j = 0; j < element_size; ++j) {
    output[global_id * element_size + j] = input[input_offset * element_size + j];
  }
}
