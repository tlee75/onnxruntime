
#define DEFINE_NEG_KERNEL(type, type_suffix) \
  __kernel void neg_##type_suffix(           \
      __global type* input,                  \
      __global type* output) {               \
    int gid = get_global_id(0);              \
    output[gid] = -input[gid];               \
  }

DEFINE_NEG_KERNEL(float, float)
#if defined(cl_khr_fp64)
DEFINE_NEG_KERNEL(double, double)
#endif
// fp16 kernels are compiled only into programs that need them (USE_FP16 is set
// for a session that has fp16 tensors, or when use_fp16 asks for fp16 maths).
// They cannot simply ride along on every program: the fp16 prelude enables
// cl_khr_fp16 for the whole program, and on Adreno that makes the *fp32*
// kernels several times slower.
#if defined(USE_FP16)
DEFINE_NEG_KERNEL(half, fp16)
#endif
