// Simple operations macros
#define ADD_OP(a, b) ((a) + (b))
#define SUB_OP(a, b) ((a) - (b))
#define MUL_OP(a, b) ((a) * (b))
#define DIV_OP(a, b) ((a) / (b))
#define POW_OP(a, b) (pow((a), (b)))

// Kernel definition macro with broadcasting support for up to 5 dimensions
#define DEFINE_OP_KERNEL(type, type_suffix, op_name, operation)           \
  __kernel void op_name##_##type_suffix(                                  \
      __global type* A,                                                   \
      __global type* B,                                                   \
      __global type* C,                                                   \
      __global int* Ashapes,                                              \
      __global int* Bshapes,                                              \
      __global int* Cshapes,                                              \
      int Adims,                                                          \
      int Bdims,                                                          \
      int Cdims) {                                                        \
    int global_id = get_global_id(0);                                     \
    int indices[5] = {0, 0, 0, 0, 0};                                     \
    int div = global_id;                                                  \
                                                                          \
    /* Calculate the index for each dimension in C (output tensor). */    \
    /* The shape values are element counts and fit int32. Doing this       */ \
    /* div/mod in 64-bit is ~10x slower on Adreno's OpenCL compiler.       */ \
    for (int i = Cdims - 1; i >= 0; --i) {                                \
      int c_dim = Cshapes[i];                                             \
      indices[i] = div % c_dim;                                           \
      div /= c_dim;                                                       \
    }                                                                     \
                                                                          \
    /* Map the broadcasted indices to A and B */                          \
    int A_idx = 0, B_idx = 0;                                             \
    int A_stride = 1, B_stride = 1;                                       \
                                                                          \
    /* Iterate from the highest dimension down to the lowest */           \
    for (int i = Cdims - 1; i >= 0; --i) {                                \
      int ashape_id = Adims - Cdims + i;                                  \
      int A_dim_size = (ashape_id < 0) ? 1 : Ashapes[ashape_id];          \
      int bshape_id = Bdims - Cdims + i;                                  \
      int B_dim_size = (bshape_id < 0) ? 1 : Bshapes[bshape_id];          \
      int idx = indices[i];                                               \
                                                                          \
      /* Adjust for broadcasting: only add index if dimension size > 1 */ \
      if (A_dim_size > 1) {                                               \
        A_idx += idx * A_stride;                                          \
      }                                                                   \
      if (B_dim_size > 1) {                                               \
        B_idx += idx * B_stride;                                          \
      }                                                                   \
                                                                          \
      A_stride *= A_dim_size;                                             \
      B_stride *= B_dim_size;                                             \
    }                                                                     \
                                                                          \
    /* Perform the operation with broadcasting */                         \
    C[global_id] = operation(A[A_idx], B[B_idx]);                         \
  }

// Define float type kernel for add operation
DEFINE_OP_KERNEL(float, float, add, ADD_OP)

DEFINE_OP_KERNEL(float, float, sub, SUB_OP)
DEFINE_OP_KERNEL(float, float, mul, MUL_OP)
DEFINE_OP_KERNEL(float, float, div, DIV_OP)
DEFINE_OP_KERNEL(float, float, pow, POW_OP)

// double type kernels
#if defined(cl_khr_fp64)
DEFINE_OP_KERNEL(double, double, add, ADD_OP)
DEFINE_OP_KERNEL(double, double, sub, SUB_OP)
DEFINE_OP_KERNEL(double, double, mul, MUL_OP)
DEFINE_OP_KERNEL(double, double, div, DIV_OP)
DEFINE_OP_KERNEL(double, double, pow, POW_OP)
#endif

// int type kernels
DEFINE_OP_KERNEL(int, int, add, ADD_OP)
DEFINE_OP_KERNEL(int, int, sub, SUB_OP)
DEFINE_OP_KERNEL(int, int, mul, MUL_OP)
DEFINE_OP_KERNEL(int, int, div, DIV_OP)
// DEFINE_OP_KERNEL(int, int, pow, POW_OP)

// long type kernels
DEFINE_OP_KERNEL(long, long, add, ADD_OP)
DEFINE_OP_KERNEL(long, long, sub, SUB_OP)
DEFINE_OP_KERNEL(long, long, mul, MUL_OP)
DEFINE_OP_KERNEL(long, long, div, DIV_OP)
// DEFINE_OP_KERNEL(long, long, pow, POW_OP)

// fp16 kernels are a DEVICE capability (cl_khr_fp16), not a session choice:
// an fp16 graph needs them whatever use_fp16 says.
#if defined(cl_khr_fp16)
DEFINE_OP_KERNEL(half, fp16, add, ADD_OP)
DEFINE_OP_KERNEL(half, fp16, sub, SUB_OP)
DEFINE_OP_KERNEL(half, fp16, mul, MUL_OP)
DEFINE_OP_KERNEL(half, fp16, div, DIV_OP)
DEFINE_OP_KERNEL(half, fp16, pow, POW_OP)
#endif

// Macros for comparison operations
#define GREATER_OP(a, b) ((a) > (b) ? 1 : 0)
#define EQUAL_OP(a, b) ((a) == (b) ? 1 : 0)

#define DEFINE_LOGICOP_KERNEL(type, type_suffix, op_name, operation)      \
  __kernel void op_name##_##type_suffix(                                  \
      __global type* A,                                                   \
      __global type* B,                                                   \
      __global uchar* C,                                                  \
      __global int* Ashapes,                                              \
      __global int* Bshapes,                                              \
      __global int* Cshapes,                                              \
      int Adims,                                                          \
      int Bdims,                                                          \
      int Cdims) {                                                        \
    int global_id = get_global_id(0);                                     \
    int indices[5] = {0, 0, 0, 0, 0};                                     \
    int div = global_id;                                                  \
                                                                          \
    /* Calculate the index for each dimension in C (output tensor). */    \
    /* The shape values are element counts and fit int32. Doing this       */ \
    /* div/mod in 64-bit is ~10x slower on Adreno's OpenCL compiler.       */ \
    for (int i = Cdims - 1; i >= 0; --i) {                                \
      int c_dim = Cshapes[i];                                             \
      indices[i] = div % c_dim;                                           \
      div /= c_dim;                                                       \
    }                                                                     \
                                                                          \
    /* Map the broadcasted indices to A and B */                          \
    int A_idx = 0, B_idx = 0;                                             \
    int A_stride = 1, B_stride = 1;                                       \
                                                                          \
    /* Iterate from the highest dimension down to the lowest */           \
    for (int i = Cdims - 1; i >= 0; --i) {                                \
      int ashape_id = Adims - Cdims + i;                                  \
      int A_dim_size = (ashape_id < 0) ? 1 : Ashapes[ashape_id];          \
      int bshape_id = Bdims - Cdims + i;                                  \
      int B_dim_size = (bshape_id < 0) ? 1 : Bshapes[bshape_id];          \
      int idx = indices[i];                                               \
                                                                          \
      /* Adjust for broadcasting: only add index if dimension size > 1 */ \
      if (A_dim_size > 1) {                                               \
        A_idx += idx * A_stride;                                          \
      }                                                                   \
      if (B_dim_size > 1) {                                               \
        B_idx += idx * B_stride;                                          \
      }                                                                   \
                                                                          \
      A_stride *= A_dim_size;                                             \
      B_stride *= B_dim_size;                                             \
    }                                                                     \
                                                                          \
    /* Perform the operation with broadcasting */                         \
    C[global_id] = operation(A[A_idx], B[B_idx]);                         \
  }

// float type kernels for Greater and Equal
DEFINE_LOGICOP_KERNEL(float, float, greater, GREATER_OP)
DEFINE_LOGICOP_KERNEL(float, float, equal, EQUAL_OP)

// double type kernels for Greater and Equal
#if defined(cl_khr_fp64)
DEFINE_LOGICOP_KERNEL(double, double, greater, GREATER_OP)
DEFINE_LOGICOP_KERNEL(double, double, equal, EQUAL_OP)
#endif

// int type kernels for Greater and Equal
DEFINE_LOGICOP_KERNEL(int, int, greater, GREATER_OP)
DEFINE_LOGICOP_KERNEL(int, int, equal, EQUAL_OP)

// long type kernels for Greater and Equal
DEFINE_LOGICOP_KERNEL(long, long, greater, GREATER_OP)
DEFINE_LOGICOP_KERNEL(long, long, equal, EQUAL_OP)

DEFINE_LOGICOP_KERNEL(uchar, bool, equal, EQUAL_OP)

// ---------------------------------------------------------------------------
// Fast paths for the common no-broadcast cases: identical input/output shapes,
// or a scalar operand. These kernels do NO index math, so they avoid integer
// division entirely (which is slow on some mobile GPUs even in 32-bit, e.g.
// Adreno). The host (elementwise.cc) uses one when it applies, else the
// general broadcast kernel above.
#define DEFINE_OP_FAST_KERNELS(type, type_suffix, op_name, operation)      \
  __kernel void op_name##_##type_suffix##_lin(                             \
      __global type* A, __global type* B, __global type* C) {              \
    long gid = get_global_id(0); /* no index math here, so 64-bit is free */ \
    C[gid] = operation(A[gid], B[gid]);                                    \
  }                                                                        \
  __kernel void op_name##_##type_suffix##_scalar_a(                        \
      __global type* A, __global type* B, __global type* C) {              \
    long gid = get_global_id(0); /* no index math here, so 64-bit is free */ \
    C[gid] = operation(A[0], B[gid]);                                      \
  }                                                                        \
  __kernel void op_name##_##type_suffix##_scalar_b(                        \
      __global type* A, __global type* B, __global type* C) {              \
    long gid = get_global_id(0); /* no index math here, so 64-bit is free */ \
    C[gid] = operation(A[gid], B[0]);                                      \
  }

#define DEFINE_LOGICOP_FAST_KERNELS(type, type_suffix, op_name, operation) \
  __kernel void op_name##_##type_suffix##_lin(                             \
      __global type* A, __global type* B, __global uchar* C) {             \
    long gid = get_global_id(0); /* no index math here, so 64-bit is free */ \
    C[gid] = operation(A[gid], B[gid]);                                    \
  }                                                                        \
  __kernel void op_name##_##type_suffix##_scalar_a(                        \
      __global type* A, __global type* B, __global uchar* C) {             \
    long gid = get_global_id(0); /* no index math here, so 64-bit is free */ \
    C[gid] = operation(A[0], B[gid]);                                      \
  }                                                                        \
  __kernel void op_name##_##type_suffix##_scalar_b(                        \
      __global type* A, __global type* B, __global uchar* C) {             \
    long gid = get_global_id(0); /* no index math here, so 64-bit is free */ \
    C[gid] = operation(A[gid], B[0]);                                      \
  }

DEFINE_OP_FAST_KERNELS(float, float, add, ADD_OP)
DEFINE_OP_FAST_KERNELS(float, float, sub, SUB_OP)
DEFINE_OP_FAST_KERNELS(float, float, mul, MUL_OP)
DEFINE_OP_FAST_KERNELS(float, float, div, DIV_OP)
DEFINE_OP_FAST_KERNELS(float, float, pow, POW_OP)
#if defined(cl_khr_fp64)
DEFINE_OP_FAST_KERNELS(double, double, add, ADD_OP)
DEFINE_OP_FAST_KERNELS(double, double, sub, SUB_OP)
DEFINE_OP_FAST_KERNELS(double, double, mul, MUL_OP)
DEFINE_OP_FAST_KERNELS(double, double, div, DIV_OP)
DEFINE_OP_FAST_KERNELS(double, double, pow, POW_OP)
#endif
DEFINE_OP_FAST_KERNELS(int, int, add, ADD_OP)
DEFINE_OP_FAST_KERNELS(int, int, sub, SUB_OP)
DEFINE_OP_FAST_KERNELS(int, int, mul, MUL_OP)
DEFINE_OP_FAST_KERNELS(int, int, div, DIV_OP)
DEFINE_OP_FAST_KERNELS(long, long, add, ADD_OP)
DEFINE_OP_FAST_KERNELS(long, long, sub, SUB_OP)
DEFINE_OP_FAST_KERNELS(long, long, mul, MUL_OP)
DEFINE_OP_FAST_KERNELS(long, long, div, DIV_OP)

DEFINE_LOGICOP_FAST_KERNELS(float, float, greater, GREATER_OP)
DEFINE_LOGICOP_FAST_KERNELS(float, float, equal, EQUAL_OP)
#if defined(cl_khr_fp64)
DEFINE_LOGICOP_FAST_KERNELS(double, double, greater, GREATER_OP)
DEFINE_LOGICOP_FAST_KERNELS(double, double, equal, EQUAL_OP)
#endif
DEFINE_LOGICOP_FAST_KERNELS(int, int, greater, GREATER_OP)
DEFINE_LOGICOP_FAST_KERNELS(int, int, equal, EQUAL_OP)
DEFINE_LOGICOP_FAST_KERNELS(long, long, greater, GREATER_OP)
DEFINE_LOGICOP_FAST_KERNELS(long, long, equal, EQUAL_OP)
DEFINE_LOGICOP_FAST_KERNELS(uchar, bool, equal, EQUAL_OP)

// fp16 kernels are a DEVICE capability (cl_khr_fp16), not a session choice:
// an fp16 graph needs them whatever use_fp16 says.
#if defined(cl_khr_fp16)
DEFINE_OP_FAST_KERNELS(half, fp16, add, ADD_OP)
DEFINE_OP_FAST_KERNELS(half, fp16, sub, SUB_OP)
DEFINE_OP_FAST_KERNELS(half, fp16, mul, MUL_OP)
DEFINE_OP_FAST_KERNELS(half, fp16, div, DIV_OP)
DEFINE_OP_FAST_KERNELS(half, fp16, pow, POW_OP)
#endif
