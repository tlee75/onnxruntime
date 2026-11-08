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
// Used by Conv (im2col + GEMM) and ConvTranspose (GEMM + col2im). The naive
// Gemm_Float above is a scalar loop with no data reuse; this one stages a
// GEMM_BM x GEMM_BK tile of A and a GEMM_BK x GEMM_BN tile of B in local
// memory and computes a GEMM_TM x GEMM_TN register tile per work-item.
//
// Two implementations, chosen by the EP's `use_fp16` flag (see the conv hosts):
//   Gemm_Tiled      - fp32 local tiles, fp32 MAC + fp32 accumulate (default).
//   Gemm_Tiled_F16  - half local tiles, packed half2 MAC, half2 accumulate.
//
// Both grid-stride over the N tiles: the grid is capped to the device's
// CL_DEVICE_MAX_WORK_ITEM_SIZES[0] (Mali tolerates a huge per-dimension global
// work size; Adreno does not, and schedules it pathologically), so a work-group
// may own several N tiles. The host launches
// (min(numNtiles, cap/(GEMM_BN/GEMM_TN)) * (GEMM_BN/GEMM_TN), numMtiles * (GEMM_BM/GEMM_TM)).
//
// Tried and rejected: a no-local-memory register variant (~3x slower), padded
// local tiles + #pragma unroll (~1.4x slower), and an fp16 variant that flushed
// to an fp32 accumulator every GEMM_BK step (~1.8x slower than the half2 form).
//
// Tile sizes are the one device-tuned choice; the algorithms are generic.
#define GEMM_BM 64
#define GEMM_BN 64
#define GEMM_BK 32
#define GEMM_TM 4
#define GEMM_TN 4

__kernel __attribute__((reqd_work_group_size(GEMM_BN / GEMM_TN, GEMM_BM / GEMM_TM, 1)))
void Gemm_Tiled(
    const int M,
    const int N,
    const int K,
    const int has_bias,
    const int trans_a,
    __global const float* A,
    __global const float* B,
    __global float* C,
    __global const float* Bias) {
  const int lx = get_local_id(0);  // N dimension
  const int ly = get_local_id(1);  // M dimension
  const int tid = ly * get_local_size(0) + lx;
  const int nthreads = get_local_size(0) * get_local_size(1);

  __local float As[GEMM_BK][GEMM_BM + 1];  // transposed (+1 pad: avoids store bank conflicts)
  __local float Bs[GEMM_BK][GEMM_BN];

  const int row0 = get_group_id(1) * GEMM_BM;
  // Bias seeds the accumulator (added once per output element); owners that
  // apply it elsewhere (ConvTranspose's col2im) pass has_bias = 0.
  float bias_r[GEMM_TM];
  for (int i = 0; i < GEMM_TM; ++i) {
    const int gr = row0 + ly * GEMM_TM + i;
    bias_r[i] = (has_bias && gr < M) ? Bias[gr] : 0.0f;
  }
  const int numNtiles = (N + GEMM_BN - 1) / GEMM_BN;

  for (int nt = get_group_id(0); nt < numNtiles; nt += get_num_groups(0)) {
    const int col0 = nt * GEMM_BN;

    float acc[GEMM_TM][GEMM_TN];
    for (int i = 0; i < GEMM_TM; ++i)
      for (int j = 0; j < GEMM_TN; ++j)
        acc[i][j] = bias_r[i];

    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
      for (int idx = tid; idx < GEMM_BM * GEMM_BK; idx += nthreads) {
        // trans_a: A is [K, M] (a ConvTranspose weight), read as A[gk*M + gr]
        // instead of A[gr*K + gk]. Either way the tile is stored transposed, so
        // mapping consecutive threads to consecutive `i` keeps the global read
        // coalesced in both layouts -- which is what lets ConvTranspose skip its
        // per-run weight-transpose kernel.
        const int i = trans_a ? (idx % GEMM_BM) : (idx / GEMM_BK);
        const int kk = trans_a ? (idx / GEMM_BM) : (idx % GEMM_BK);
        const int gr = row0 + i;
        const int gk = k0 + kk;
        As[kk][i] = (gr < M && gk < K) ? A[trans_a ? (gk * M + gr) : (gr * K + gk)] : 0.0f;
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
        for (int i = 0; i < GEMM_TM; ++i) a[i] = As[kk][ly * GEMM_TM + i];
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
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// fp16 variant: half local tiles, packed half2 MAC and half2 accumulate. The
// half2 accumulator is converted to fp32 only when the result is stored, so the
// whole reduction accumulates in fp16.
__kernel __attribute__((reqd_work_group_size(GEMM_BN / GEMM_TN, GEMM_BM / GEMM_TM, 1)))
void Gemm_Tiled_F16(
    const int M,
    const int N,
    const int K,
    const int has_bias,
    const int trans_a,
    __global const float* A,
    __global const float* B,
    __global float* C,
    __global const float* Bias) {
  const int lx = get_local_id(0);
  const int ly = get_local_id(1);
  const int tid = ly * get_local_size(0) + lx;
  const int nthreads = get_local_size(0) * get_local_size(1);

  // +2 (one 4-byte bank) so that filling As[i][kk] with consecutive threads on
  // consecutive `i` (the trans_a / ConvTranspose layout) does not land every
  // thread in the same local-memory bank.
  __local half As[GEMM_BM][GEMM_BK + 2];
  __local half Bs[GEMM_BK][GEMM_BN];

  const int row0 = get_group_id(1) * GEMM_BM;
  half bias_r[GEMM_TM];
  for (int i = 0; i < GEMM_TM; ++i) {
    const int gr = row0 + ly * GEMM_TM + i;
    bias_r[i] = (half)((has_bias && gr < M) ? Bias[gr] : 0.0f);
  }
  const int numNtiles = (N + GEMM_BN - 1) / GEMM_BN;

  for (int nt = get_group_id(0); nt < numNtiles; nt += get_num_groups(0)) {
    const int col0 = nt * GEMM_BN;

    half2 acc2[GEMM_TM][GEMM_TN / 2];
    for (int i = 0; i < GEMM_TM; ++i)
      for (int j2 = 0; j2 < GEMM_TN / 2; ++j2)
        acc2[i][j2] = (half2)(bias_r[i], bias_r[i]);

    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
      for (int idx = tid; idx < GEMM_BM * GEMM_BK; idx += nthreads) {
        const int i = trans_a ? (idx % GEMM_BM) : (idx / GEMM_BK);
        const int kk = trans_a ? (idx / GEMM_BM) : (idx % GEMM_BK);
        const int gr = row0 + i;
        const int gk = k0 + kk;
        As[i][kk] = (half)((gr < M && gk < K) ? A[trans_a ? (gk * M + gr) : (gr * K + gk)] : 0.0f);
      }
      for (int idx = tid; idx < GEMM_BK * GEMM_BN; idx += nthreads) {
        const int kk = idx / GEMM_BN;
        const int j = idx % GEMM_BN;
        const int gk = k0 + kk;
        const int gc = col0 + j;
        Bs[kk][j] = (half)((gk < K && gc < N) ? B[gk * N + gc] : 0.0f);
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      for (int kk = 0; kk < GEMM_BK; ++kk) {
        half a[GEMM_TM];
        for (int i = 0; i < GEMM_TM; ++i) a[i] = As[ly * GEMM_TM + i][kk];
        for (int j2 = 0; j2 < GEMM_TN / 2; ++j2) {
          const int jc = lx * GEMM_TN + 2 * j2;
          const half2 b2 = (half2)(Bs[kk][jc], Bs[kk][jc + 1]);
          for (int i = 0; i < GEMM_TM; ++i)
            acc2[i][j2] = (half2)(a[i], a[i]) * b2 + acc2[i][j2];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < GEMM_TM; ++i) {
      const int gr = row0 + ly * GEMM_TM + i;
      if (gr >= M) continue;
      for (int j2 = 0; j2 < GEMM_TN / 2; ++j2) {
        const int gc = col0 + lx * GEMM_TN + 2 * j2;
        if (gc < N) C[gr * N + gc] = (float)acc2[i][j2].lo;
        if (gc + 1 < N) C[gr * N + gc + 1] = (float)acc2[i][j2].hi;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif
