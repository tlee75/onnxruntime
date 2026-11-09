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
    __global const float* Bias,
    const int sa, const int sb, const int sc) {
  // Batching (used by the Winograd paths): plane p = get_group_id(2) selects the
  // p-th operand plane. The plain 2-D grid is plane 0 with zero offsets.
  const int plane = get_group_id(2);
  A += (size_t)plane * sa;
  B += (size_t)plane * sb;
  C += (size_t)plane * sc;
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
    __global const float* Bias,
    const int sa, const int sb, const int sc) {
  const int plane = get_group_id(2);
  A += (size_t)plane * sa;
  B += (size_t)plane * sb;
  C += (size_t)plane * sc;
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

// Same tiling and the same half2 accumulate as Gemm_Tiled_F16, but the operands
// and the result are half *in memory* -- the storage strategy of an fp16 graph
// (and of LiteRT's GPU delegate). The fp32-storage F16 variant above converts each
// element on load and on store; here the bytes moved are already half.
__kernel __attribute__((reqd_work_group_size(GEMM_BN / GEMM_TN, GEMM_BM / GEMM_TM, 1)))
void Gemm_Tiled_F16H(
    const int M,
    const int N,
    const int K,
    const int has_bias,
    const int trans_a,
    __global const half* A,
    __global const half* B,
    __global half* C,
    __global const half* Bias,
    const int sa, const int sb, const int sc) {
  // DEV-ONLY batching (used by the Winograd paths): plane p = get_group_id(2)
  // selects the p-th operand plane, so ONE launch covers all six F(4,3) transform
  // points and gets a 6x bigger grid. A 2D grid means plane 0 and zero offsets,
  // i.e. the plain GEMM.
  const int plane = get_group_id(2);
  A += (size_t)plane * sa;
  B += (size_t)plane * sb;
  C += (size_t)plane * sc;
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
    bias_r[i] = (has_bias && gr < M) ? Bias[gr] : (half)0;
  }
  const int numNtiles = (N + GEMM_BN - 1) / GEMM_BN;

  for (int nt = get_group_id(0); nt < numNtiles; nt += get_num_groups(0)) {
    const int col0 = nt * GEMM_BN;

    half2 acc2[GEMM_TM][GEMM_TN / 2];
    for (int i = 0; i < GEMM_TM; ++i)
      for (int j2 = 0; j2 < GEMM_TN / 2; ++j2)
        acc2[i][j2] = (half2)(bias_r[i], bias_r[i]);

    for (int k0 = 0; k0 < K; k0 += GEMM_BK) {
      // Packed half2 global loads where the addresses are structurally contiguous
      // AND 4-byte aligned: one 4-byte access per two elements. A scalar 2-byte
      // global access is disproportionately slow on Adreno -- this half-storage
      // twin measured ~8% slower than the fp32-storage Gemm_Tiled_F16 despite
      // moving half the bytes -- so pairing the elements is the point. The guards
      // are RUNTIME, so an odd stride or an under-aligned tensor takes the scalar
      // path with no second kernel variant (and no host-side capability query).
      const bool a_pack = ((((size_t)A) & 3) == 0) &&
                          ((trans_a ? (M & 1) : (K & 1)) == 0);
      if (!trans_a) {
        if (a_pack) {
          for (int idx = tid; idx < GEMM_BM * (GEMM_BK / 2); idx += nthreads) {
            const int i = idx / (GEMM_BK / 2);
            const int kp = (idx % (GEMM_BK / 2)) * 2;
            const int gr = row0 + i;
            const int gk = k0 + kp;
            if (gr < M && gk + 1 < K) {
              const half2 v = *(__global const half2*)(A + gr * K + gk);
              As[i][kp] = v.lo;
              As[i][kp + 1] = v.hi;
            } else {
              As[i][kp] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0;
              As[i][kp + 1] = (gr < M && gk + 1 < K) ? A[gr * K + gk + 1] : (half)0;
            }
          }
        } else {
          for (int idx = tid; idx < GEMM_BM * GEMM_BK; idx += nthreads) {
            const int i = idx / GEMM_BK;
            const int kk = idx % GEMM_BK;
            const int gr = row0 + i;
            const int gk = k0 + kk;
            As[i][kk] = (gr < M && gk < K) ? A[gr * K + gk] : (half)0;
          }
        }
      } else {
        if (a_pack) {
          for (int idx = tid; idx < GEMM_BK * (GEMM_BM / 2); idx += nthreads) {
            const int kk = idx / (GEMM_BM / 2);
            const int ip = (idx % (GEMM_BM / 2)) * 2;
            const int gr = row0 + ip;
            const int gk = k0 + kk;
            if (gk < K && gr + 1 < M) {
              const half2 v = *(__global const half2*)(A + gk * M + gr);
              As[ip][kk] = v.lo;
              As[ip + 1][kk] = v.hi;
            } else {
              As[ip][kk] = (gr < M && gk < K) ? A[gk * M + gr] : (half)0;
              As[ip + 1][kk] = (gr + 1 < M && gk < K) ? A[gk * M + gr + 1] : (half)0;
            }
          }
        } else {
          for (int idx = tid; idx < GEMM_BM * GEMM_BK; idx += nthreads) {
            const int i = idx % GEMM_BM;
            const int kk = idx / GEMM_BM;
            const int gr = row0 + i;
            const int gk = k0 + kk;
            As[i][kk] = (gr < M && gk < K) ? A[gk * M + gr] : (half)0;
          }
        }
      }
      if ((((size_t)B) & 3) == 0 && (N & 1) == 0) {
        for (int idx = tid; idx < GEMM_BK * (GEMM_BN / 2); idx += nthreads) {
          const int kk = idx / (GEMM_BN / 2);
          const int jp = (idx % (GEMM_BN / 2)) * 2;
          const int gk = k0 + kk;
          const int gc = col0 + jp;
          if (gk < K && gc + 1 < N) {
            const half2 v = *(__global const half2*)(B + gk * N + gc);
            Bs[kk][jp] = v.lo;
            Bs[kk][jp + 1] = v.hi;
          } else {
            Bs[kk][jp] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0;
            Bs[kk][jp + 1] = (gk < K && gc + 1 < N) ? B[gk * N + gc + 1] : (half)0;
          }
        }
      } else {
        for (int idx = tid; idx < GEMM_BK * GEMM_BN; idx += nthreads) {
          const int kk = idx / GEMM_BN;
          const int j = idx % GEMM_BN;
          const int gk = k0 + kk;
          const int gc = col0 + j;
          Bs[kk][j] = (gk < K && gc < N) ? B[gk * N + gc] : (half)0;
        }
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

    if ((((size_t)C) & 3) == 0 && (N & 1) == 0) {
      for (int i = 0; i < GEMM_TM; ++i) {
        const int gr = row0 + ly * GEMM_TM + i;
        if (gr >= M) continue;
        for (int j2 = 0; j2 < GEMM_TN / 2; ++j2) {
          const int gc = col0 + lx * GEMM_TN + 2 * j2;
          if (gc + 1 < N) {
            *(__global half2*)(C + gr * N + gc) = acc2[i][j2];
          } else if (gc < N) {
            C[gr * N + gc] = acc2[i][j2].lo;
          }
        }
      }
    } else {
      for (int i = 0; i < GEMM_TM; ++i) {
        const int gr = row0 + ly * GEMM_TM + i;
        if (gr >= M) continue;
        for (int j2 = 0; j2 < GEMM_TN / 2; ++j2) {
          const int gc = col0 + lx * GEMM_TN + 2 * j2;
          if (gc < N) C[gr * N + gc] = acc2[i][j2].lo;
          if (gc + 1 < N) C[gr * N + gc + 1] = acc2[i][j2].hi;
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}
#endif

// ---------------------------------------------------------------------------
// DEV-ONLY (ORT_OPENCL_EXP=winograd / winoct): 1D Winograd F(4,3) transforms.
// They live in gemm.cl because BOTH conv.cl and convtranspose.cl include this
// file, so one definition serves both programs. If this ever ships, move them to
// nn/kernels/winograd.cl and include that from both.
//
//   U[t][m][row] = sum_rr G[t][rr] * W[m][row*3 + rr]        (weights, computed once)
//   V[t][row][j] = sum_rr Bt[t][rr] * X[c][4j - pad + 3s + rr]
//   P[t]         = U[t] @ V[t]                               (Gemm_Tiled_F16H)
//   Y[m][os*(4j+q)+r] = bias[m] + sum_t At[q][t] * P[t][m][j]
// with row = s*C + c, s in [0, NS) a 3-tap slice of the original kernel (K=3 is
// NS=1, K=7 is NS=3 with the third slice zero-padded; a ConvTranspose PHASE is
// NS=1 with two real taps and a zero third).
#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void WinoWeightT(__global const half* Wt, const int M, const int C,
                          const int K, const int NS, const int t,
                          const int u_stride, __global half* U) {
  const int row = get_global_id(0);  // s*C + c
  const int m = get_global_id(1);
  const int NC = NS * C;
  if (m >= M || row >= NC) return;
  const int s = row / C;
  const int c = row - s * C;
  const int b = m * (C * K) + c * K + 3 * s;
  const float g0 = (3 * s + 0 < K) ? (float)Wt[b] : 0.0f;
  const float g1 = (3 * s + 1 < K) ? (float)Wt[b + 1] : 0.0f;
  const float g2 = (3 * s + 2 < K) ? (float)Wt[b + 2] : 0.0f;
  float u;
  if (t == 0) u = 0.25f * g0;
  else if (t == 1) u = -(g0 + g1 + g2) / 6.0f;
  else if (t == 2) u = (-g0 + g1 - g2) / 6.0f;
  else if (t == 3) u = g0 / 24.0f + g1 / 12.0f + g2 / 6.0f;
  else if (t == 4) u = g0 / 24.0f - g1 / 12.0f + g2 / 6.0f;
  else u = g2;
  U[t * u_stride + m * NC + row] = (half)u;
}

__kernel void WinoInputT(__global const half* X, const int C, const int W,
                         const int NS, const int pad, const int T, const int t,
                         const int v_stride, __global half* V) {
  const int j = get_global_id(0);
  const int row = get_global_id(1);  // s*C + c
  const int NC = NS * C;
  if (j >= T || row >= NC) return;
  const int s = row / C;
  const int c = row - s * C;
  const int base = 4 * j - pad + 3 * s;
  float d[6];
  for (int r = 0; r < 6; ++r) {
    const int iw = base + r;
    d[r] = (iw >= 0 && iw < W) ? (float)X[c * W + iw] : 0.0f;
  }
  float v;
  if (t == 0) v = 4.0f * d[0] - 5.0f * d[2] + d[4];
  else if (t == 1) v = -4.0f * d[1] - 4.0f * d[2] + d[3] + d[4];
  else if (t == 2) v = 4.0f * d[1] - 4.0f * d[2] - d[3] + d[4];
  else if (t == 3) v = -2.0f * d[1] - d[2] + 2.0f * d[3] + d[4];
  else if (t == 4) v = 2.0f * d[1] - d[2] - 2.0f * d[3] + d[4];
  else v = 4.0f * d[1] - 5.0f * d[3] + d[5];
  V[t * v_stride + row * T + j] = (half)v;
}

// os/r place the tile in the output: os=1,r=0 for a plain Conv, and os=stride,
// r=phase for a ConvTranspose phase (which writes every os-th element). P holds
// the six transform planes back to back, p_stride apart.
__kernel void WinoOutputT(__global const half* P, const int p_stride,
                          const int M, const int T, const int has_bias,
                          __global const half* B, __global half* Y, const int Wout,
                          const int os, const int r) {
  const int j = get_global_id(0);
  const int m = get_global_id(1);
  if (j >= T || m >= M) return;
  const int o = m * T + j;
  const float p0 = (float)P[0 * p_stride + o], p1 = (float)P[1 * p_stride + o];
  const float p2 = (float)P[2 * p_stride + o], p3 = (float)P[3 * p_stride + o];
  const float p4 = (float)P[4 * p_stride + o], p5 = (float)P[5 * p_stride + o];
  const float bias = has_bias ? (float)B[m] : 0.0f;
  const float y0 = p0 + p1 + p2 + p3 + p4;
  const float y1 = p1 - p2 + 2.0f * p3 - 2.0f * p4;
  const float y2 = p1 + p2 + 4.0f * p3 + 4.0f * p4;
  const float y3 = p1 - p2 + 8.0f * p3 - 8.0f * p4 + p5;
  const int q0 = 4 * j;
  const int ow0 = os * q0 + r;
  if (ow0 < Wout) Y[m * Wout + ow0] = (half)(y0 + bias);
  if (os * (q0 + 1) + r < Wout) Y[m * Wout + os * (q0 + 1) + r] = (half)(y1 + bias);
  if (os * (q0 + 2) + r < Wout) Y[m * Wout + os * (q0 + 2) + r] = (half)(y2 + bias);
  if (os * (q0 + 3) + r < Wout) Y[m * Wout + os * (q0 + 3) + r] = (half)(y3 + bias);
}

// DEV-ONLY (ORT_OPENCL_EXP=winoct): gather ONE ConvTranspose phase's two taps --
// k0 = (r+pad) mod s and k0+s of the [C][M*K] weight -- into a [M][C][3] filter
// [w(k0+s), w(k0), 0] that the F(4,3) pipeline can consume. The phase window
// starts at 4q + a0 with a0 = (r + pad - k0)/s, so as a 3-tap correlation its pad
// is 1 - a0.
__kernel void WinoCtGather(__global const half* Wt, const int M, const int C,
                           const int K, const int s, const int k0,
                           __global half* Wp) {
  const int c = get_global_id(0);
  const int m = get_global_id(1);
  if (m >= M || c >= C) return;
  const int base = c * (M * K) + m * K + k0;
  // g[0] multiplies x[q + a0 - 1], which comes from tap k0+s; g[1] multiplies
  // x[q + a0], from tap k0 (pad3 = 1 - a0 puts those at rr = 0 and 1).
  const float g0 = (k0 + s < K) ? (float)Wt[base + s] : 0.0f;  // tap k0+s
  const float g1 = (k0 < K) ? (float)Wt[base] : 0.0f;          // tap k0
  const int o = (m * C + c) * 3;
  Wp[o + 0] = (half)g0;
  Wp[o + 1] = (half)g1;
  Wp[o + 2] = (half)0;
}
#endif
