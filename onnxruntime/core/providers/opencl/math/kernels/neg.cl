
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
// fp16 kernels are a DEVICE capability (cl_khr_fp16), not a session choice:
// an fp16 graph needs them whatever use_fp16 says.
#if defined(cl_khr_fp16)
DEFINE_NEG_KERNEL(half, fp16)
#endif
