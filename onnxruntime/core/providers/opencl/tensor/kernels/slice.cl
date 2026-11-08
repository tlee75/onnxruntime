// Shapes/strides are int32 element counts and the per-dim index math is 32-bit:
// the per-dim values fit int for any tensor the EP accepts, and 64-bit integer
// division is pathologically slow on some mobile GPUs (e.g. Adreno). The
// *flattened* size can still exceed int for a very large tensor, so the kernel
// is generated twice: `Slice` is the always-correct 64-bit form and
// `Slice_I32` is the fast path the host takes only when the byte size provably
// fits int. Only the running offsets/addresses widen (the per-dim metadata stays
// int), so the 64-bit form is correct up to 2^63 elements and pays the slow
// divides only in that (rare) case.
#define DEFINE_SLICE_KERNEL(NAME, IDX)                                         \
  __kernel void NAME(__global uchar* input,                                    \
                     __global uchar* output,                                   \
                     int input_dimensions,                                     \
                     int element_size,                                         \
                     __global int* input_starts,                               \
                     __global int* input_steps,                                \
                     __global int* input_strides,                              \
                     __global int* output_strides,                             \
                     int output_dimensions,                                    \
                     long max_input_size) {                                    \
    IDX gid = get_global_id(0);                                                \
    IDX output_idx = gid;                                                      \
    IDX input_offset = 0;                                                      \
                                                                               \
    for (int i = 0; i < output_dimensions; ++i) {                              \
      IDX coord = output_idx / output_strides[i];                              \
      output_idx %= output_strides[i];                                         \
                                                                               \
      IDX input_coord = input_starts[i] + coord * input_steps[i];              \
      input_offset += input_coord * input_strides[i];                          \
      if (input_offset * element_size >= max_input_size) {                     \
        return;                                                                \
      }                                                                        \
    }                                                                          \
                                                                               \
    for (int j = 0; j < element_size; ++j) {                                   \
      output[gid * element_size + j] = input[input_offset * element_size + j]; \
    }                                                                          \
  }

// `Slice` is the ALWAYS-correct form (64-bit); `Slice_I32` is the fast path the
// host picks only when the byte size provably fits int. That way a host mistake
// degrades to slow, never to wrong.
DEFINE_SLICE_KERNEL(Slice, long)
DEFINE_SLICE_KERNEL(Slice_I32, int)
