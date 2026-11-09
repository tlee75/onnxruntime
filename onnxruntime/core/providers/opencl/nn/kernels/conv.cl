// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Reuse the tiled GEMM (Gemm_Tiled) from the math kernels.
#include "math/kernels/gemm.cl"

// 1D convolution (cross-correlation), matching the ONNX Conv operator.
// X:   [N, C, W]        (row-major)
// Wt:  [M, C/group, K]
// B:   [M] (optional)
// Y:   [N, M, Wout], Wout = floor((W + pad_l + pad_r - dil*(K-1) - 1)/stride) + 1
//
// Naive fallback (one work-item per output, scalar inner loop). The fast path
// is im2col (Conv_Im2col) + Gemm_Tiled, used for group==1, batch==1.
__kernel void Conv1D_Float(
    __global const float* X,
    __global const float* Wt,
    __global const float* B,
    __global float* Y,
    const int N,
    const int C,
    const int W,
    const int M,
    const int K,
    const int Wout,
    const int group,
    const int stride,
    const int pad,
    const int dilation,
    const int has_bias) {
  int gid = get_global_id(0);
  if (gid >= N * M * Wout) {
    return;
  }
  int ow = gid % Wout;
  int t = gid / Wout;
  int m = t % M;
  int n = t / M;

  int Mg = M / group;
  int Cg = C / group;
  int g = m / Mg;
  int cbase = g * Cg;

  float acc = has_bias ? B[m] : 0.0f;
  for (int c = 0; c < Cg; ++c) {
    __global const float* xrow = X + (n * C + cbase + c) * W;
    __global const float* wrow = Wt + (m * Cg + c) * K;
    for (int k = 0; k < K; ++k) {
      int iw = ow * stride - pad + k * dilation;
      if (iw >= 0 && iw < W) {
        acc += xrow[iw] * wrow[k];
      }
    }
  }
  Y[gid] = acc;
}

// im2col for the fast path (group==1, one batch): unfold X [C, W] into
// col [C*K, Wout], where col[c*K + k, ow] = X[c, ow*stride - pad + k*dil]
// (zero outside the input). Then Y[M, Wout] = W[M, C*K] @ col.
//
// Materializing the unfold once is cheaper than gathering it inside the GEMM's
// B-tile load: the B tile is re-read M/GEMM_BM times, and a gather (with its
// index math and bounds checks) is slower per element than a coalesced read of
// a contiguous col — measured faster even though col costs an extra write.
//
// gws = (min(Wout, device cap), C*K). The `ow` axis GRID-STRIDES so the global
// work size never exceeds CL_DEVICE_MAX_WORK_ITEM_SIZES[0] — Mali tolerates a
// huge per-dimension size, Adreno does not (and schedules it pathologically).
__kernel void Conv_Im2col(
    const int C,
    const int W,
    const int K,
    const int Wout,
    const int stride,
    const int pad,
    const int dil,
    __global const float* X,
    __global float* col) {
  const int j = get_global_id(1);  // c*K + k
  if (j >= C * K) return;
  const int c = j / K;
  const int k = j - c * K;
  const int tap = k * dil;
  const int g0 = get_global_size(0);
  for (int ow = get_global_id(0); ow < Wout; ow += g0) {
    const int iw = ow * stride - pad + tap;
    col[j * Wout + ow] = (iw >= 0 && iw < W) ? X[c * W + iw] : 0.0f;
  }
}

// Add the per-output-channel bias to a Conv result: Y[m, ow] += B[m].
// gws = (min(Wout, device cap), M); the `ow` axis grid-strides (see above).
// Fused conv GEMM: Y[M, Wout] = Wt[M, C*K] @ im2col(X[C, W]), with the im2col
// "B" operand GATHERED inline from X instead of materializing a col buffer.
//
// Templated on the tile so the host can pick a small-BM variant: the conv
// GEMMs are SHORT and WIDE (M is tiny -- down to 1 -- while Wout is huge), so
// a BM x BN accumulator tile wastes (BM - M) padded rows when M < BM.
#define DEFINE_GEMM_TILED_CONV(NAME, BM, BN, BK, TM, TN)                        \
  __kernel __attribute__((reqd_work_group_size(BN / TN, BM / TM, 1)))           \
  void NAME(                                                                    \
      const int M, const int Wout, const int C, const int W, const int K,       \
      const int stride, const int pad, const int dil,                           \
      const int has_bias,                                                       \
      __global const float* Wt, __global const float* X, __global float* Y,     \
      __global const float* B) {                                                \
    const int lx = get_local_id(0);                                             \
    const int ly = get_local_id(1);                                             \
    const int tid = ly * get_local_size(0) + lx;                                \
    const int nthreads = get_local_size(0) * get_local_size(1);                 \
    __local float As[BK][BM + 1]; /* transposed; +1 pad avoids store bank conflicts */                                                   \
    __local float Bs[BK][BN];                                                   \
    const int row0 = get_group_id(1) * BM;                                      \
    const int Kdim = C * K;                                                     \
    /* Seed the accumulator with the bias: adds it exactly once per output      \
       element, so no separate bias pass over Y is needed. */                    \
    float bias_r[TM];                                                           \
    for (int i = 0; i < TM; ++i) {                                              \
      const int gr = row0 + ly * TM + i;                                        \
      bias_r[i] = (has_bias && gr < M) ? B[gr] : 0.0f;                          \
    }                                                                           \
    const int numNtiles = (Wout + BN - 1) / BN;                                 \
    for (int nt = get_group_id(0); nt < numNtiles; nt += get_num_groups(0)) {   \
      const int col0 = nt * BN;                                                 \
      float acc[TM][TN];                                                        \
      for (int i = 0; i < TM; ++i)                                              \
        for (int j = 0; j < TN; ++j) acc[i][j] = bias_r[i];                     \
      for (int k0 = 0; k0 < Kdim; k0 += BK) {                                   \
        for (int idx = tid; idx < BM * BK; idx += nthreads) {                   \
          const int i = idx / BK;                                               \
          const int kk = idx % BK;                                              \
          const int gr = row0 + i;                                              \
          const int gk = k0 + kk;                                               \
          As[kk][i] = (gr < M && gk < Kdim) ? Wt[gr * Kdim + gk] : 0.0f;        \
        }                                                                       \
        for (int idx = tid; idx < BK * BN; idx += nthreads) {                   \
          const int kk = idx / BN;                                              \
          const int j = idx % BN;                                               \
          const int gk = k0 + kk;                                               \
          const int ow = col0 + j;                                              \
          float v = 0.0f;                                                       \
          if (gk < Kdim && ow < Wout) {                                         \
            const int c = gk / K;                                               \
            const int tap = gk - c * K;                                         \
            const int iw = ow * stride - pad + tap * dil;                       \
            if (iw >= 0 && iw < W) v = X[c * W + iw];                           \
          }                                                                     \
          Bs[kk][j] = v;                                                        \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
        for (int kk = 0; kk < BK; ++kk) {                                       \
          float a[TM];                                                          \
          float b[TN];                                                          \
          for (int i = 0; i < TM; ++i) a[i] = As[kk][ly * TM + i];              \
          for (int j = 0; j < TN; ++j) b[j] = Bs[kk][lx * TN + j];              \
          for (int i = 0; i < TM; ++i)                                          \
            for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];              \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
      }                                                                         \
      for (int i = 0; i < TM; ++i) {                                            \
        const int gr = row0 + ly * TM + i;                                      \
        if (gr >= M) continue;                                                  \
        for (int j = 0; j < TN; ++j) {                                          \
          const int gc = col0 + lx * TN + j;                                    \
          if (gc < Wout) Y[gr * Wout + gc] = acc[i][j];                         \
        }                                                                       \
      }                                                                         \
      barrier(CLK_LOCAL_MEM_FENCE);                                             \
    }                                                                           \
  }

// The host picks by M (roughly: BM should be >= M but as small as possible).
DEFINE_GEMM_TILED_CONV(Gemm_Tiled_Conv, 64, 64, 32, 4, 4)
DEFINE_GEMM_TILED_CONV(Gemm_Tiled_Conv_BM32, 32, 64, 32, 4, 4)
DEFINE_GEMM_TILED_CONV(Gemm_Tiled_Conv_BM16, 16, 64, 32, 4, 4)
// Narrower BN (=> 2x the N-tiles, i.e. more work-groups) for the short-and-
// narrow conv GEMMs (small Wout), where BN=64 leaves the grid under-filled.
DEFINE_GEMM_TILED_CONV(Gemm_Tiled_Conv_BN32, 64, 32, 32, 4, 4)

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// fp16 twin of DEFINE_GEMM_TILED_CONV: half local tiles (=> half the local
// footprint, so more work-groups fit — the reason fp16 Gemm_Tiled is far
// faster than its fp32 form on Adreno) and half2 MAC/accumulate. The global
// tensors stay fp32; only the tiles are narrowed. Selected by the host when the
// device advertises fp16 AND the session asked for it (addOpenCL(useFp16,..)).
#define DEFINE_GEMM_TILED_CONV_F16(NAME, BM, BN, BK, TM, TN)                     \
  __kernel __attribute__((reqd_work_group_size(BN / TN, BM / TM, 1)))           \
  void NAME(const int M, const int Wout, const int C, const int W,              \
            const int K, const int stride, const int pad, const int dil,        \
            const int has_bias,                                                 \
            __global const float* Wt, __global const float* X,                   \
            __global float* Y, __global const float* B) {                       \
    const int lx = get_local_id(0);                                             \
    const int ly = get_local_id(1);                                             \
    const int tid = ly * get_local_size(0) + lx;                                \
    const int nthreads = get_local_size(0) * get_local_size(1);                 \
    __local half As[BM][BK];                                                    \
    __local half Bs[BK][BN];                                                    \
    const int row0 = get_group_id(1) * BM;                                      \
    const int Kdim = C * K;                                                     \
    half bias_r[TM];                                                            \
    for (int i = 0; i < TM; ++i) {                                              \
      const int gr = row0 + ly * TM + i;                                        \
      bias_r[i] = (half)((has_bias && gr < M) ? B[gr] : 0.0f);                  \
    }                                                                           \
    const int numNtiles = (Wout + BN - 1) / BN;                                 \
    for (int nt = get_group_id(0); nt < numNtiles; nt += get_num_groups(0)) {   \
      const int col0 = nt * BN;                                                 \
      half2 acc2[TM][TN / 2];                                                    \
      for (int i = 0; i < TM; ++i)                                              \
        for (int j2 = 0; j2 < TN / 2; ++j2)                                     \
          acc2[i][j2] = (half2)(bias_r[i], bias_r[i]);                          \
      for (int k0 = 0; k0 < Kdim; k0 += BK) {                                   \
        for (int idx = tid; idx < BM * BK; idx += nthreads) {                   \
          const int i = idx / BK;                                               \
          const int kk = idx % BK;                                              \
          const int gr = row0 + i;                                              \
          const int gk = k0 + kk;                                               \
          As[i][kk] = (half)((gr < M && gk < Kdim) ? Wt[gr * Kdim + gk] : 0.0f);\
        }                                                                       \
        for (int idx = tid; idx < BK * BN; idx += nthreads) {                   \
          const int kk = idx / BN;                                              \
          const int j = idx % BN;                                               \
          const int gk = k0 + kk;                                               \
          const int ow = col0 + j;                                              \
          float v = 0.0f;                                                       \
          if (gk < Kdim && ow < Wout) {                                         \
            const int c = gk / K;                                               \
            const int tap = gk - c * K;                                         \
            const int iw = ow * stride - pad + tap * dil;                       \
            if (iw >= 0 && iw < W) v = X[c * W + iw];                           \
          }                                                                     \
          Bs[kk][j] = (half)v;                                                  \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
        for (int kk = 0; kk < BK; ++kk) {                                       \
          half a[TM];                                                           \
          for (int i = 0; i < TM; ++i) a[i] = As[ly * TM + i][kk];              \
          for (int j2 = 0; j2 < TN / 2; ++j2) {                                 \
            const int jc = lx * TN + 2 * j2;                                    \
            const half2 b2 = (half2)(Bs[kk][jc], Bs[kk][jc + 1]);               \
            for (int i = 0; i < TM; ++i)                                        \
              acc2[i][j2] = (half2)(a[i], a[i]) * b2 + acc2[i][j2];             \
          }                                                                     \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
      }                                                                         \
      for (int i = 0; i < TM; ++i) {                                            \
        const int gr = row0 + ly * TM + i;                                      \
        if (gr >= M) continue;                                                  \
        for (int j2 = 0; j2 < TN / 2; ++j2) {                                   \
          const int gc = col0 + lx * TN + 2 * j2;                               \
          if (gc < Wout) Y[gr * Wout + gc] = (float)acc2[i][j2].lo;             \
          if (gc + 1 < Wout) Y[gr * Wout + gc + 1] = (float)acc2[i][j2].hi;     \
        }                                                                       \
      }                                                                         \
      barrier(CLK_LOCAL_MEM_FENCE);                                             \
    }                                                                           \
  }

DEFINE_GEMM_TILED_CONV_F16(Gemm_Tiled_Conv_F16, 64, 64, 32, 4, 4)
DEFINE_GEMM_TILED_CONV_F16(Gemm_Tiled_Conv_BM32_F16, 32, 64, 32, 4, 4)
DEFINE_GEMM_TILED_CONV_F16(Gemm_Tiled_Conv_BM16_F16, 16, 64, 32, 4, 4)
DEFINE_GEMM_TILED_CONV_F16(Gemm_Tiled_Conv_BN32_F16, 64, 32, 32, 4, 4)

// Half-*storage* twins: same tiles and half2 accumulate, but Wt/X/Y/B are half in
// memory -- what an fp16 graph (and the GPU delegate) actually hands us. The F16
// family above keeps fp32 storage and converts per element on load and store.
#define DEFINE_GEMM_TILED_CONV_H(NAME, BM, BN, BK, TM, TN, NG, PB)             \
  __kernel __attribute__((reqd_work_group_size(BN / TN, BM / TM, 1)))           \
  void NAME(const int M, const int Wout, const int C, const int W,              \
            const int K, const int stride, const int pad, const int dil,        \
            const int has_bias,                                                 \
            __global const half* Wt, __global const half* X,                   \
            __global half* Y, __global const half* B) {                       \
    const int lx = get_local_id(0);                                             \
    const int ly = get_local_id(1);                                             \
    const int tid = ly * get_local_size(0) + lx;                                \
    const int nthreads = get_local_size(0) * get_local_size(1);                 \
    __local half As[BM][BK];                                                    \
    /* B is declared as half2 so a PB read is ONE 32-bit LDS op instead of two  */ \
    /* 16-bit reads plus a pack; the scalar store path aliases it as half (the   */ \
    /* same bytes, so the footprint is unchanged).                              */ \
    __local half2 BsP[BK][BN / 2];                                              \
    /* 2D half view of the same bytes, so the scalar store path still says      */ \
    /* Bs[kk][j] and the base stays half2-aligned (4 bytes).                    */ \
    __local half(*Bs)[BN] = (__local half(*)[BN])BsP;                           \
    const int row0 = get_group_id(1) * BM;                                      \
    const int Kdim = C * K;                                                     \
    half bias_r[TM];                                                            \
    for (int i = 0; i < TM; ++i) {                                              \
      const int gr = row0 + ly * TM + i;                                        \
      bias_r[i] = ((has_bias && gr < M) ? B[gr] : (half)0);                  \
    }                                                                           \
    const int numNtiles = (Wout + BN - 1) / BN;                                 \
    for (int nt = get_group_id(0); nt < numNtiles; nt += get_num_groups(0)) {   \
      const int col0 = nt * BN;                                                 \
      half2 acc2[TM][TN / 2];                                                    \
      for (int i = 0; i < TM; ++i)                                              \
        for (int j2 = 0; j2 < TN / 2; ++j2)                                     \
          acc2[i][j2] = (half2)(bias_r[i], bias_r[i]);                          \
      for (int k0 = 0; k0 < Kdim; k0 += BK) {                                   \
        if ((((size_t)Wt) & 3) == 0 && (Kdim & 1) == 0) {                      \
          for (int idx = tid; idx < BM * (BK / 2); idx += nthreads) {           \
            const int i = idx / (BK / 2);                                       \
            const int kp = (idx % (BK / 2)) * 2;                               \
            const int gr = row0 + i;                                            \
            const int gk = k0 + kp;                                             \
            if (gr < M && gk + 1 < Kdim) {                                      \
              const half2 v = *(__global const half2*)(Wt + gr * Kdim + gk);    \
              As[i][kp] = v.lo;                                                 \
              As[i][kp + 1] = v.hi;                                             \
            } else {                                                            \
              As[i][kp] = ((gr < M && gk < Kdim) ? Wt[gr * Kdim + gk] : (half)0); \
              As[i][kp + 1] = ((gr < M && gk + 1 < Kdim) ? Wt[gr * Kdim + gk + 1] : (half)0); \
            }                                                                   \
          }                                                                     \
        } else {                                                                \
          for (int idx = tid; idx < BM * BK; idx += nthreads) {                 \
            const int i = idx / BK;                                             \
            const int kk = idx % BK;                                            \
            const int gr = row0 + i;                                            \
            const int gk = k0 + kk;                                             \
            As[i][kk] = ((gr < M && gk < Kdim) ? Wt[gr * Kdim + gk] : (half)0); \
          }                                                                     \
        }                                                                       \
        for (int idx = tid; idx < BK * BN; idx += nthreads) {                   \
          const int kk = idx / BN;                                              \
          const int j = idx % BN;                                               \
          const int gk = k0 + kk;                                               \
          const int ow = col0 + j;                                              \
          half v = (half)0;                                                     \
          if (gk < Kdim && ow < Wout) {                                         \
            if (NG) {                                                           \
              /* DIAGNOSTIC (ORT_OPENCL_EXP=conv_nogather): read X CONTIGUOUSLY */ \
              /* instead of via the strided im2col index. Wrong on purpose --    */ \
              /* this exists only to price the gather with the paired harness.  */ \
              v = X[(gk % C) * W + (ow % W)];                                   \
            } else {                                                            \
              const int c = gk / K;                                             \
              const int tap = gk - c * K;                                       \
              const int iw = ow * stride - pad + tap * dil;                     \
              if (iw >= 0 && iw < W) v = X[c * W + iw];                         \
            }                                                                   \
          }                                                                     \
          Bs[kk][j] = v;                                                        \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
        for (int kk = 0; kk < BK; ++kk) {                                       \
          half a[TM];                                                           \
          for (int i = 0; i < TM; ++i) a[i] = As[ly * TM + i][kk];              \
          for (int j2 = 0; j2 < TN / 2; ++j2) {                                 \
            const int jc = lx * TN + 2 * j2;                                    \
            const half2 b2 = PB ? BsP[kk][jc / 2] : (half2)(Bs[kk][jc], Bs[kk][jc + 1]); \
            for (int i = 0; i < TM; ++i)                                        \
              acc2[i][j2] = (half2)(a[i], a[i]) * b2 + acc2[i][j2];             \
          }                                                                     \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
      }                                                                         \
      if ((((size_t)Y) & 3) == 0 && (Wout & 1) == 0) {                          \
        for (int i = 0; i < TM; ++i) {                                          \
          const int gr = row0 + ly * TM + i;                                    \
          if (gr >= M) continue;                                                \
          for (int j2 = 0; j2 < TN / 2; ++j2) {                                 \
            const int gc = col0 + lx * TN + 2 * j2;                             \
            if (gc + 1 < Wout) {                                                \
              *(__global half2*)(Y + gr * Wout + gc) = acc2[i][j2];             \
            } else if (gc < Wout) {                                             \
              Y[gr * Wout + gc] = acc2[i][j2].lo;                               \
            }                                                                   \
          }                                                                     \
        }                                                                       \
      } else {                                                                  \
        for (int i = 0; i < TM; ++i) {                                          \
          const int gr = row0 + ly * TM + i;                                    \
          if (gr >= M) continue;                                                \
          for (int j2 = 0; j2 < TN / 2; ++j2) {                                 \
            const int gc = col0 + lx * TN + 2 * j2;                             \
            if (gc < Wout) Y[gr * Wout + gc] = acc2[i][j2].lo;             \
            if (gc + 1 < Wout) Y[gr * Wout + gc + 1] = acc2[i][j2].hi;     \
          }                                                                     \
        }                                                                       \
      }                                                                         \
      barrier(CLK_LOCAL_MEM_FENCE);                                             \
    }                                                                           \
  }


DEFINE_GEMM_TILED_CONV_H(Gemm_Tiled_Conv_H, 64, 64, 32, 4, 4, 0, 0)
DEFINE_GEMM_TILED_CONV_H(Gemm_Tiled_Conv_BM32_H, 32, 64, 32, 4, 4, 0, 0)
DEFINE_GEMM_TILED_CONV_H(Gemm_Tiled_Conv_BM16_H, 16, 64, 32, 4, 4, 0, 0)
DEFINE_GEMM_TILED_CONV_H(Gemm_Tiled_Conv_BN32_H, 64, 32, 32, 4, 4, 0, 0)
// DEV-ONLY experiment: the same tile with a larger register block (TM=TN=8, so a
// 8x8 work-group and 4x the outputs per work-item). Selected by ORT_OPENCL_EXP so
// the bench's paired A/B can run both in one process. NOT for the upstream series.
DEFINE_GEMM_TILED_CONV_H(Gemm_Tiled_Conv_H_TM8, 64, 64, 32, 8, 8, 0, 0)
// DIAGNOSTIC (ORT_OPENCL_EXP=conv_nogather): contiguous read, WRONG output, to
// price the strided im2col gather. Dev-only; never for the upstream series.
DEFINE_GEMM_TILED_CONV_H(Gemm_Tiled_Conv_H_NG, 64, 64, 32, 4, 4, 1, 0)
// A/B (ORT_OPENCL_EXP=conv_packb): read B as ONE packed half2 LDS op instead of
// two 16-bit reads plus a pack (the per-FMA-overhead hypothesis). Dev-only.
DEFINE_GEMM_TILED_CONV_H(Gemm_Tiled_Conv_H_PB, 64, 64, 32, 4, 4, 0, 1)

// DEV-ONLY (ORT_OPENCL_EXP=image2d): the same half-storage fused conv, but the
// activation X is bound as a CL_RGBA/CL_HALF_FLOAT image2d -- 4 consecutive columns
// per texel -- and the im2col read becomes a texture SAMPLE instead of a buffer
// gather. The host creates/caches the image and copies X into it each call. This
// exists ONLY to price the texture path against the buffer path with the paired
// harness; it is not for the upstream series. Note the confound: this variant
// keeps the plain A load and Y store (no packed variants), so it is ~a few percent
// behind the production kernel before the B load is even considered.
#define DEFINE_GEMM_TILED_CONV_H_IMG(NAME, BM, BN, BK, TM, TN)                  \
  __kernel __attribute__((reqd_work_group_size(BN / TN, BM / TM, 1)))           \
  void NAME(const int M, const int Wout, const int C, const int W,              \
            const int K, const int stride, const int pad, const int dil,        \
            const int has_bias,                                                 \
            __global const half* Wt, __read_only image2d_t X,                   \
            __global half* Y, __global const half* B) {                         \
    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; \
    const int lx = get_local_id(0);                                             \
    const int ly = get_local_id(1);                                             \
    const int tid = ly * get_local_size(0) + lx;                                \
    const int nthreads = get_local_size(0) * get_local_size(1);                 \
    __local half As[BM][BK];                                                    \
    __local half Bs[BK][BN];                                                    \
    const int row0 = get_group_id(1) * BM;                                      \
    const int Kdim = C * K;                                                     \
    half bias_r[TM];                                                            \
    for (int i = 0; i < TM; ++i) {                                              \
      const int gr = row0 + ly * TM + i;                                        \
      bias_r[i] = ((has_bias && gr < M) ? B[gr] : (half)0);                     \
    }                                                                           \
    const int numNtiles = (Wout + BN - 1) / BN;                                 \
    for (int nt = get_group_id(0); nt < numNtiles; nt += get_num_groups(0)) {   \
      const int col0 = nt * BN;                                                 \
      half2 acc2[TM][TN / 2];                                                    \
      for (int i = 0; i < TM; ++i)                                              \
        for (int j2 = 0; j2 < TN / 2; ++j2)                                     \
          acc2[i][j2] = (half2)(bias_r[i], bias_r[i]);                          \
      for (int k0 = 0; k0 < Kdim; k0 += BK) {                                   \
        for (int idx = tid; idx < BM * BK; idx += nthreads) {                   \
          const int i = idx / BK;                                               \
          const int kk = idx % BK;                                              \
          const int gr = row0 + i;                                              \
          const int gk = k0 + kk;                                               \
          As[i][kk] = ((gr < M && gk < Kdim) ? Wt[gr * Kdim + gk] : (half)0);   \
        }                                                                       \
        for (int idx = tid; idx < BK * BN; idx += nthreads) {                   \
          const int kk = idx / BN;                                              \
          const int j = idx % BN;                                               \
          const int gk = k0 + kk;                                               \
          const int ow = col0 + j;                                              \
          half v = (half)0;                                                     \
          if (gk < Kdim && ow < Wout) {                                         \
            const int c = gk / K;                                               \
            const int tap = gk - c * K;                                         \
            const int iw = ow * stride - pad + tap * dil;                       \
            if (iw >= 0 && iw < W) {                                            \
              const half4 t = read_imageh(X, smp, (int2)(iw >> 2, c));          \
              switch (iw & 3) {                                                 \
                case 0: v = t.x; break;                                         \
                case 1: v = t.y; break;                                         \
                case 2: v = t.z; break;                                         \
                default: v = t.w; break;                                        \
              }                                                                 \
            }                                                                   \
          }                                                                     \
          Bs[kk][j] = v;                                                        \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
        for (int kk = 0; kk < BK; ++kk) {                                       \
          half a[TM];                                                           \
          for (int i = 0; i < TM; ++i) a[i] = As[ly * TM + i][kk];              \
          for (int j2 = 0; j2 < TN / 2; ++j2) {                                 \
            const int jc = lx * TN + 2 * j2;                                    \
            const half2 b2 = (half2)(Bs[kk][jc], Bs[kk][jc + 1]);               \
            for (int i = 0; i < TM; ++i)                                        \
              acc2[i][j2] = (half2)(a[i], a[i]) * b2 + acc2[i][j2];             \
          }                                                                     \
        }                                                                       \
        barrier(CLK_LOCAL_MEM_FENCE);                                           \
      }                                                                         \
      for (int i = 0; i < TM; ++i) {                                            \
        const int gr = row0 + ly * TM + i;                                      \
        if (gr >= M) continue;                                                  \
        for (int j2 = 0; j2 < TN / 2; ++j2) {                                   \
          const int gc = col0 + lx * TN + 2 * j2;                               \
          if (gc < Wout) Y[gr * Wout + gc] = acc2[i][j2].lo;                    \
          if (gc + 1 < Wout) Y[gr * Wout + gc + 1] = acc2[i][j2].hi;            \
        }                                                                       \
      }                                                                         \
      barrier(CLK_LOCAL_MEM_FENCE);                                             \
    }                                                                           \
  }

// The A/B arm: the same tile, texture-sampled X.
DEFINE_GEMM_TILED_CONV_H_IMG(Gemm_Tiled_Conv_H_IMG, 64, 64, 32, 4, 4)

// DEV-ONLY (ORT_OPENCL_EXP=image2d): fill the CL_RGBA/CL_HALF_FLOAT image from X
// ([C, W] halfs, 4 consecutive columns per texel). Done with a kernel rather than
// clEnqueueCopyBufferToImage because the latter assumes the buffer's row pitch
// matches the image's (implementation-chosen) pitch, which our tightly packed X
// does not guarantee.
__kernel void ImgFillX(__global const half* X, const int C, const int W,
                       __write_only image2d_t Dst) {
  const int x = get_global_id(0);
  const int c = get_global_id(1);
  if (c >= C || x * 4 >= W) return;
  const int iw = x * 4;
  half4 v = (half4)((half)0, (half)0, (half)0, (half)0);
  v.x = X[c * W + iw];
  if (iw + 1 < W) v.y = X[c * W + iw + 1];
  if (iw + 2 < W) v.z = X[c * W + iw + 2];
  if (iw + 3 < W) v.w = X[c * W + iw + 3];
  write_imageh(Dst, (int2)(x, c), v);
}
#endif
