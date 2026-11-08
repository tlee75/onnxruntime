__kernel void Gemm_Float(
    const int trans_a,
    const int trans_b,
    const long M,
    const long N,
    const long K,
    __global float* A,
    __global float* B,
    __global float* C,
    __global long* Cshapes,
    __global float* Output,
    const int C_exist) {
  int row = get_global_id(0);
  int col = get_global_id(1);

  if (row < M && col < N) {
    float sum = 0.0;

    for (int k = 0; k < K; ++k) {
      float a_val = (trans_a == 0) ? A[row * K + k] : A[k * M + row];
      float b_val = (trans_b == 0) ? B[k * N + col] : B[col * K + k];
      sum += a_val * b_val;
    }

    if (C_exist) {
      int C_rows = Cshapes[0];
      int C_cols = Cshapes[1];
      if (C_rows == M && C_cols == N) {
        sum += C[row * N + col];
      } else if (C_rows == 1 && C_cols == N) {
        sum += C[col];
      } else if (C_rows == M && C_cols == 1) {
        sum += C[row];
      } else if (C_rows == 1 && C_cols == 1) {
        sum += C[0];
      }
    }

    Output[row * N + col] = sum;
  }
}

// ---------------------------------------------------------------------------
// Tiled, register-blocked GEMM: C[M,N] = A[M,K] @ B[K,N] (both row-major).
//
// Used by Conv (im2col + GEMM) and ConvTranspose (GEMM + col2im), which is
// where the vocoder spends its time. The naive Gemm_Float above is a scalar
// loop with no data reuse; this one stages a GEMM_BM x GEMM_BK tile of A and a
// GEMM_BK x GEMM_BN tile of B in local memory and computes a GEMM_TM x GEMM_TN
// register tile per work-item, so each loaded element feeds many FMAs.
//
// Tried and rejected (measured on the A51 Mali-G76, all slower than this form):
// a no-local-memory register variant (~2x slower — __local reuse matters), and
// padding the local tiles against bank conflicts + #pragma unroll (~1.4x
// slower — the extra __local hurt more than the conflicts).
//
// Layout: work-item (lx,ly) owns the GEMM_TN columns at col0+lx*GEMM_TN and the
// GEMM_TM rows at row0+ly*GEMM_TM; the tile loads are coalesced. The host passes
// the matching local work size (GEMM_BN/GEMM_TN, GEMM_BM/GEMM_TM) and pads the
// global range to a multiple of it.
//
// NOTE: these tile sizes are tuned for Mali (the A51 G76) and are compile-time
// constants, so they are the one device-specific choice here. The algorithms
// (im2col/GEMM/col2im) are generic. If this is used on Adreno, prefer selecting
// the tile by GpuType (the EP already models MALI vs ADRENO) rather than a new
// EP name.
#define GEMM_BM 64
#define GEMM_BN 64
#define GEMM_BK 16
#define GEMM_TM 4
#define GEMM_TN 4

__kernel __attribute__((reqd_work_group_size(GEMM_BN / GEMM_TN, GEMM_BM / GEMM_TM, 1)))
void Gemm_Tiled(
    const int M,
    const int N,
    const int K,
    __global const float* A,
    __global const float* B,
    __global float* C) {
  const int lx = get_local_id(0);  // N dimension
  const int ly = get_local_id(1);  // M dimension
  const int tid = ly * get_local_size(0) + lx;
  const int nthreads = get_local_size(0) * get_local_size(1);

  __local float As[GEMM_BM][GEMM_BK];
  __local float Bs[GEMM_BK][GEMM_BN];

  const int row0 = get_group_id(1) * GEMM_BM;
  const int col0 = get_group_id(0) * GEMM_BN;

  float acc[GEMM_TM][GEMM_TN];
  for (int i = 0; i < GEMM_TM; ++i)
    for (int j = 0; j < GEMM_TN; ++j)
      acc[i][j] = 0.0f;

  for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
    for (int idx = tid; idx < GEMM_BM * GEMM_BK; idx += nthreads) {
      const int i = idx / GEMM_BK;
      const int kk = idx % GEMM_BK;
      const int gr = row0 + i;
      const int gk = k0 + kk;
      As[i][kk] = (gr < M && gk < K) ? A[gr * K + gk] : 0.0f;
    }
    for (int idx = tid; idx < GEMM_BK * GEMM_BN; idx += nthreads) {
      const int kk = idx / GEMM_BN;
      const int j = idx % GEMM_BN;
      const int gk = k0 + kk;
      const int gc = col0 + j;
      Bs[kk][j] = (gk < K && gc < N) ? B[gk * N + gc] : 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int kk = 0; kk < GEMM_BK; ++kk) {
      float a[GEMM_TM];
      float b[GEMM_TN];
      for (int i = 0; i < GEMM_TM; ++i) a[i] = As[ly * GEMM_TM + i][kk];
      for (int j = 0; j < GEMM_TN; ++j) b[j] = Bs[kk][lx * GEMM_TN + j];
      for (int i = 0; i < GEMM_TM; ++i)
        for (int j = 0; j < GEMM_TN; ++j)
          acc[i][j] += a[i] * b[j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (int i = 0; i < GEMM_TM; ++i) {
    const int gr = row0 + ly * GEMM_TM + i;
    if (gr >= M) continue;
    for (int j = 0; j < GEMM_TN; ++j) {
      const int gc = col0 + lx * GEMM_TN + j;
      if (gc < N) C[gr * N + gc] = acc[i][j];
    }
  }
}
