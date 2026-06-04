// fdtd_cuda.cu — Native CUDA kernels for the FDTD time loop.
//
// This file is the GPU port of:
//   * assembly/fdtd_kernels.ts        (updateH / updateE bulk)
//   * src/cpml.js                     (6-face CPML magnetic + electric updates)
//   * src/sources.js                  (voltage source injection)
//   * src/sampling.js                 (sampled voltage / current reduction)
//
// CONTRACT (src/backends/CONTRACT.md) is the single source of truth. The index
// math and per-step update order here are a byte-for-byte transliteration of the
// reference JS. All math is f32 on the device (the reference is f64; validation
// allows f32 tolerance per CONTRACT §1).
//
// Memory model: every field array, coefficient array, CPML b/a/Psi/CPsi array,
// and source index/coeff array is cudaMalloc'd ONCE in fdtd_init/upload and
// stays resident for the whole run. Per step we only:
//   - launch the bulk + CPML + injection kernels,
//   - run a small reduction for the sampled voltage/current scalars.
// Per batch we copy back the accumulated per-step sample scalars.
//
// The host-facing C API is declared in fdtd_cuda.h and consumed by binding.cpp.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "fdtd_cuda.h"

// ───────────────────────── Error handling ──────────────────────────────────
#define CUDA_OK(call) do {                                              \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        std::snprintf(g_lastError, sizeof(g_lastError),                 \
            "CUDA error %s at %s:%d -> %s",                             \
            #call, __FILE__, __LINE__, cudaGetErrorString(_e));        \
        return FDTD_ERR;                                                \
    }                                                                   \
} while (0)

static char g_lastError[512] = {0};
const char* fdtd_last_error() { return g_lastError; }

// ───────────────────────── Device state ────────────────────────────────────
// One global solver context. The JS driver runs a single problem at a time.
namespace {

struct CpmlFaceDev {
    int active = 0;
    int nc = 0;
    // 1D coefficient arrays (length nc)
    float* b_e = nullptr; float* a_e = nullptr;
    float* b_m = nullptr; float* a_m = nullptr;
    // Psi (convolution) + CPsi (coefficient) arrays. Names mirror cpml.js.
    // Only the arrays relevant to a face-axis are allocated; the rest stay null.
    float* Psi0 = nullptr; float* CPsi0 = nullptr; // first  (Hy/Hz/...) per axis
    float* Psi1 = nullptr; float* CPsi1 = nullptr; // second
    float* Psi2 = nullptr; float* CPsi2 = nullptr; // first  E
    float* Psi3 = nullptr; float* CPsi3 = nullptr; // second E
    // index bookkeeping
    int m_start = 0;  // H-field start index along the face axis (0-based)
    int e_start = 0;  // E-field correction start index (0-based)
    int ascending = 0;
    // per-array element counts
    int n0 = 0, n1 = 0, n2 = 0, n3 = 0;
};

struct SolverDev {
    int inited = 0;
    int nx, ny, nz, nxp1, nyp1, nzp1;
    int nhx, nhy, nhz, nex, ney, nez;
    float dx, dy, dz, dt;

    // Fields
    float *Hx, *Hy, *Hz, *Ex, *Ey, *Ez;
    // Coefficients (18)
    float *Cexe, *Cexhz, *Cexhy;
    float *Ceye, *Ceyhx, *Ceyhz;
    float *Ceze, *Cezhy, *Cezhx;
    float *Chxh, *Chxey, *Chxez;
    float *Chyh, *Chyez, *Chyex;
    float *Chzh, *Chzex, *Chzey;

    // CPML faces: xn, xp, yn, yp, zn, zp
    CpmlFaceDev cpml[6];

    // Voltage source (single, default problem). dir: 0=x,1=y,2=z.
    int vsActive = 0;
    int vsDir = 0;
    int vsCount = 0;             // number of field indices
    int* vsIdx = nullptr;        // device int[vsCount]
    float* vsCoef = nullptr;     // device float[vsCount] (Cexs/Ceys/Cezs)
    float* vsWave = nullptr;     // device float[numberOfTimeSteps] voltage_per_e_field

    // Sampled voltage (single). Reduction target.
    int svActive = 0;
    int svDir = 0;
    int svCount = 0;
    int* svIdx = nullptr;        // device int[svCount]
    float* svUnitW = nullptr;    // device float[svCount] all 1.0 (gather weights)
    float svCsvf = 0.0f;
    float* svPartial = nullptr;  // device scratch for reduction
    float* svHostPinned = nullptr;

    // Sampled current (single). Uses a precomputed term list:
    //   sum_n sign_n * step_n * H?[idx_n]
    // We precompute (idx, weight) pairs on the host where
    //   weight = (+/- d?) * (overall direction sign).
    int siActive = 0;
    int siCount = 0;
    int* siHxIdx = nullptr;  float* siHxW = nullptr;  int siHxN = 0;
    int* siHyIdx = nullptr;  float* siHyW = nullptr;  int siHyN = 0;
    int* siHzIdx = nullptr;  float* siHzW = nullptr;  int siHzN = 0;
    float* siPartial = nullptr;

    int numberOfTimeSteps = 0;
};

SolverDev g = {};

inline int blocks1D(int n, int tpb) { return (n + tpb - 1) / tpb; }

} // anonymous namespace

// ════════════════════════════ BULK KERNELS ═════════════════════════════════
// One thread per field cell. Index math identical to fdtd_kernels.ts.

__global__ void k_updateHx(int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        float* Hx, const float* Ey, const float* Ez,
        const float* Chxh, const float* Chxey, const float* Chxez) {
    // Hx over i in [0,nxp1), j in [0,ny), k in [0,nz)
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i = blockIdx.z*blockDim.z + threadIdx.z;
    if (i>=nxp1 || j>=ny || k>=nz) return;
    int idx = i*ny*nz + j*nz + k;                  // idxHx
    int eyOff = i*ny*nzp1 + j*nzp1;                 // Ey (nxp1,ny,nzp1)
    float eey1 = Ey[eyOff + k + 1];
    float eey0 = Ey[eyOff + k];
    int ezOff0 = i*nyp1*nz + j*nz;                  // Ez (nxp1,nyp1,nz)
    int ezOff1 = i*nyp1*nz + (j+1)*nz;
    float eez1 = Ez[ezOff1 + k];
    float eez0 = Ez[ezOff0 + k];
    Hx[idx] = Chxh[idx]*Hx[idx] + Chxey[idx]*(eey1-eey0) + Chxez[idx]*(eez1-eez0);
}

__global__ void k_updateHy(int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        float* Hy, const float* Ez, const float* Ex,
        const float* Chyh, const float* Chyez, const float* Chyex) {
    // Hy over i in [0,nx), j in [0,nyp1), k in [0,nz)
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i = blockIdx.z*blockDim.z + threadIdx.z;
    if (i>=nx || j>=nyp1 || k>=nz) return;
    int idx = i*nyp1*nz + j*nz + k;                 // idxHy
    int ezOff0 = i*nyp1*nz + j*nz;                  // Ez (nxp1,nyp1,nz): i and i+1
    int ezOff1 = (i+1)*nyp1*nz + j*nz;
    float eez1 = Ez[ezOff1 + k];
    float eez0 = Ez[ezOff0 + k];
    int exOff = i*nyp1*nzp1 + j*nzp1;               // Ex (nx,nyp1,nzp1): k and k+1
    float eex1 = Ex[exOff + k + 1];
    float eex0 = Ex[exOff + k];
    Hy[idx] = Chyh[idx]*Hy[idx] + Chyez[idx]*(eez1-eez0) + Chyex[idx]*(eex1-eex0);
}

__global__ void k_updateHz(int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        float* Hz, const float* Ex, const float* Ey,
        const float* Chzh, const float* Chzex, const float* Chzey) {
    // Hz over i in [0,nx), j in [0,ny), k in [0,nzp1)
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i = blockIdx.z*blockDim.z + threadIdx.z;
    if (i>=nx || j>=ny || k>=nzp1) return;
    int idx = i*ny*nzp1 + j*nzp1 + k;               // idxHz
    int exOff0 = i*nyp1*nzp1 + j*nzp1;              // Ex (nx,nyp1,nzp1): j and j+1
    int exOff1 = i*nyp1*nzp1 + (j+1)*nzp1;
    float eex1 = Ex[exOff1 + k];
    float eex0 = Ex[exOff0 + k];
    int eyOff0 = i*ny*nzp1 + j*nzp1;                // Ey (nxp1,ny,nzp1): i and i+1
    int eyOff1 = (i+1)*ny*nzp1 + j*nzp1;
    float eey1 = Ey[eyOff1 + k];
    float eey0 = Ey[eyOff0 + k];
    Hz[idx] = Chzh[idx]*Hz[idx] + Chzex[idx]*(eex1-eex0) + Chzey[idx]*(eey1-eey0);
}

__global__ void k_updateEx(int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        float* Ex, const float* Hz, const float* Hy,
        const float* Cexe, const float* Cexhz, const float* Cexhy) {
    // Ex over i in [0,nx), j in [1,ny), k in [1,nz)
    int k = blockIdx.x*blockDim.x + threadIdx.x;    // 0-based; we add 1 below
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i = blockIdx.z*blockDim.z + threadIdx.z;
    k += 1; j += 1;
    if (i>=nx || j>=ny || k>=nz) return;
    int idx = i*nyp1*nzp1 + j*nzp1 + k;             // idxEx
    int hzBase = i*ny*nzp1;                          // Hz (nx,ny,nzp1)
    float hhz1 = Hz[hzBase + j*nzp1 + k];
    float hhz0 = Hz[hzBase + (j-1)*nzp1 + k];
    int hyOff = i*nyp1*nz + j*nz;                    // Hy (nx,nyp1,nz)
    float hhy1 = Hy[hyOff + k];
    float hhy0 = Hy[hyOff + k - 1];
    Ex[idx] = Cexe[idx]*Ex[idx] + Cexhz[idx]*(hhz1-hhz0) + Cexhy[idx]*(hhy1-hhy0);
}

__global__ void k_updateEy(int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        float* Ey, const float* Hx, const float* Hz,
        const float* Ceye, const float* Ceyhx, const float* Ceyhz) {
    // Ey over i in [1,nxp1), j in [0,ny), k in [1,nz)
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i = blockIdx.z*blockDim.z + threadIdx.z;
    k += 1; i += 1;
    if (i>=nxp1 || j>=ny || k>=nz) return;
    int idx = i*ny*nzp1 + j*nzp1 + k;               // idxEy
    int hxOff = i*ny*nz + j*nz;                      // Hx (nxp1,ny,nz)
    float hhx1 = Hx[hxOff + k];
    float hhx0 = Hx[hxOff + k - 1];
    int hzOff0 = (i-1)*ny*nzp1 + j*nzp1;            // Hz (nx,ny,nzp1): i-1 and i
    int hzOff1 = i*ny*nzp1 + j*nzp1;
    float hhz1 = Hz[hzOff1 + k];
    float hhz0 = Hz[hzOff0 + k];
    Ey[idx] = Ceye[idx]*Ey[idx] + Ceyhx[idx]*(hhx1-hhx0) + Ceyhz[idx]*(hhz1-hhz0);
}

__global__ void k_updateEz(int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        float* Ez, const float* Hy, const float* Hx,
        const float* Ceze, const float* Cezhy, const float* Cezhx) {
    // Ez over i in [1,nxp1), j in [1,nyp1), k in [0,nz)
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int i = blockIdx.z*blockDim.z + threadIdx.z;
    j += 1; i += 1;
    if (i>=nxp1 || j>=nyp1 || k>=nz) return;
    int idx = i*nyp1*nz + j*nz + k;                 // idxEz
    int hyOff0 = (i-1)*nyp1*nz + j*nz;             // Hy (nx,nyp1,nz): i-1 and i
    int hyOff1 = i*nyp1*nz + j*nz;
    float hhy1 = Hy[hyOff1 + k];
    float hhy0 = Hy[hyOff0 + k];
    int hxBase = i*ny*nz;                            // Hx (nxp1,ny,nz): j-1 and j
    float hhx1 = Hx[hxBase + j*nz + k];
    float hhx0 = Hx[hxBase + (j-1)*nz + k];
    Ez[idx] = Ceze[idx]*Ez[idx] + Cezhy[idx]*(hhy1-hhy0) + Cezhx[idx]*(hhx1-hhx0);
}

// ════════════════════════════ CPML KERNELS ═════════════════════════════════
// Each face is handled in two phases (matching cpml.js):
//   phase A: update Psi convolution arrays
//   phase B: add CPsi*Psi correction onto the H or E field
// Because phase B reads Psi written in phase A, we launch them as separate
// kernels (a kernel boundary is a device-wide sync).
//
// We dispatch x/y/z faces with dedicated kernels to keep index math explicit.

// ── X faces (xn ascending=0, xp ascending=1) ──────────────────────────────
// Magnetic: Psi_hyx (nc,nyp1,nz) from Ez; Psi_hzx (nc,ny,nzp1) from Ey.
__global__ void k_cpml_x_mag_psi(int nc,int ny,int nz,int nyp1,int nzp1,
        int eBase, // 0-based Ez/Ey i-index for ci=0 (xn:0, xp:n_st)
        const float* b_m, const float* a_m,
        const float* Ey, const float* Ez,
        float* Psi_hyx, float* Psi_hzx) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int total0 = nc*nyp1*nz;
    int total1 = nc*ny*nzp1;
    if (n < total0) {
        int ci = n / (nyp1*nz);
        int rem = n - ci*(nyp1*nz);
        int j = rem / nz;
        int k = rem - j*nz;
        int ei = eBase + ci;
        float bm = b_m[ci], am = a_m[ci];
        float ez1 = Ez[(ei+1)*nyp1*nz + j*nz + k];
        float ez0 = Ez[ei*nyp1*nz + j*nz + k];
        Psi_hyx[n] = bm*Psi_hyx[n] + am*(ez1-ez0);
    }
    if (n < total1) {
        int ci = n / (ny*nzp1);
        int rem = n - ci*(ny*nzp1);
        int j = rem / nzp1;
        int k = rem - j*nzp1;
        int ei = eBase + ci;
        float bm = b_m[ci], am = a_m[ci];
        float ey1 = Ey[(ei+1)*ny*nzp1 + j*nzp1 + k];
        float ey0 = Ey[ei*ny*nzp1 + j*nzp1 + k];
        Psi_hzx[n] = bm*Psi_hzx[n] + am*(ey1-ey0);
    }
}

__global__ void k_cpml_x_mag_corr(int nc,int ny,int nz,int nyp1,int nzp1,
        int hBase, // 0-based Hy/Hz i-index for ci=0 (= m_start)
        float* Hy, float* Hz,
        const float* CPsi_hyx, const float* Psi_hyx,
        const float* CPsi_hzx, const float* Psi_hzx) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int total0 = nc*nyp1*nz;
    int total1 = nc*ny*nzp1;
    if (n < total0) {
        int ci = n / (nyp1*nz);
        int rem = n - ci*(nyp1*nz);
        int j = rem / nz;
        int k = rem - j*nz;
        int i = hBase + ci;
        Hy[i*nyp1*nz + j*nz + k] += CPsi_hyx[n]*Psi_hyx[n];
    }
    if (n < total1) {
        int ci = n / (ny*nzp1);
        int rem = n - ci*(ny*nzp1);
        int j = rem / nzp1;
        int k = rem - j*nzp1;
        int i = hBase + ci;
        Hz[i*ny*nzp1 + j*nzp1 + k] += CPsi_hzx[n]*Psi_hzx[n];
    }
}

// Electric: Psi_eyx (nc,ny,nzp1) from Hz; Psi_ezx (nc,nyp1,nz) from Hy.
__global__ void k_cpml_x_ele_psi(int nc,int ny,int nz,int nyp1,int nzp1,
        int hBase, // Hz/Hy source i-index for ci=0 (xn:0, xp:n_st-1)
        const float* b_e, const float* a_e,
        const float* Hy, const float* Hz,
        float* Psi_eyx, float* Psi_ezx) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int total0 = nc*ny*nzp1;
    int total1 = nc*nyp1*nz;
    if (n < total0) {
        int ci = n / (ny*nzp1);
        int rem = n - ci*(ny*nzp1);
        int j = rem / nzp1;
        int k = rem - j*nzp1;
        int hi = hBase + ci;
        float be = b_e[ci], ae = a_e[ci];
        float hz1 = Hz[(hi+1)*ny*nzp1 + j*nzp1 + k];
        float hz0 = Hz[hi*ny*nzp1 + j*nzp1 + k];
        Psi_eyx[n] = be*Psi_eyx[n] + ae*(hz1-hz0);
    }
    if (n < total1) {
        int ci = n / (nyp1*nz);
        int rem = n - ci*(nyp1*nz);
        int j = rem / nz;
        int k = rem - j*nz;
        int hi = hBase + ci;
        float be = b_e[ci], ae = a_e[ci];
        float hy1 = Hy[(hi+1)*nyp1*nz + j*nz + k];
        float hy0 = Hy[hi*nyp1*nz + j*nz + k];
        Psi_ezx[n] = be*Psi_ezx[n] + ae*(hy1-hy0);
    }
}

__global__ void k_cpml_x_ele_corr(int nc,int ny,int nz,int nyp1,int nzp1,
        int eBase, // Ey/Ez correction i-index for ci=0 (= e_start)
        float* Ey, float* Ez,
        const float* CPsi_eyx, const float* Psi_eyx,
        const float* CPsi_ezx, const float* Psi_ezx) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int total0 = nc*ny*nzp1;
    int total1 = nc*nyp1*nz;
    if (n < total0) {
        int ci = n / (ny*nzp1);
        int rem = n - ci*(ny*nzp1);
        int j = rem / nzp1;
        int k = rem - j*nzp1;
        int i = eBase + ci;
        Ey[i*ny*nzp1 + j*nzp1 + k] += CPsi_eyx[n]*Psi_eyx[n];
    }
    if (n < total1) {
        int ci = n / (nyp1*nz);
        int rem = n - ci*(nyp1*nz);
        int j = rem / nz;
        int k = rem - j*nz;
        int i = eBase + ci;
        Ez[i*nyp1*nz + j*nz + k] += CPsi_ezx[n]*Psi_ezx[n];
    }
}

// ── Y faces ────────────────────────────────────────────────────────────────
// Layout (ni, nc, nk). Magnetic: Psi_hxy (nxp1,nc,nz) from Hx; Psi_hzy (nx,nc,nzp1) from Hz.
__global__ void k_cpml_y_mag_psi(int nc,int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        int jBase, // 0-based Hx/Hz j-index for ci=0 (= m_start)
        const float* b_m, const float* a_m,
        const float* Hx, const float* Hz,
        float* Psi_hxy, float* Psi_hzy) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int total0 = nxp1*nc*nz;     // Psi_hxy
    int total1 = nx*nc*nzp1;     // Psi_hzy
    if (n < total0) {
        int i = n / (nc*nz);
        int rem = n - i*(nc*nz);
        int ci = rem / nz;
        int k = rem - ci*nz;
        int j_h = jBase + ci;
        float bm = b_m[ci], am = a_m[ci];
        float hx1 = Hx[i*ny*nz + (j_h+1)*nz + k];
        float hx0 = Hx[i*ny*nz + j_h*nz + k];
        Psi_hxy[n] = bm*Psi_hxy[n] + am*(hx1-hx0);
    }
    if (n < total1) {
        int i = n / (nc*nzp1);
        int rem = n - i*(nc*nzp1);
        int ci = rem / nzp1;
        int k = rem - ci*nzp1;
        int j_h = jBase + ci;
        float bm = b_m[ci], am = a_m[ci];
        float hz1 = Hz[i*ny*nzp1 + (j_h+1)*nzp1 + k];
        float hz0 = Hz[i*ny*nzp1 + j_h*nzp1 + k];
        Psi_hzy[n] = bm*Psi_hzy[n] + am*(hz1-hz0);
    }
}

__global__ void k_cpml_y_mag_corr(int nc,int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        int jBase,
        float* Hx, float* Hz,
        const float* CPsi_hxy, const float* Psi_hxy,
        const float* CPsi_hzy, const float* Psi_hzy) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int total0 = nxp1*nc*nz;
    int total1 = nx*nc*nzp1;
    if (n < total0) {
        int i = n / (nc*nz);
        int rem = n - i*(nc*nz);
        int ci = rem / nz;
        int k = rem - ci*nz;
        int j = jBase + ci;
        Hx[i*ny*nz + j*nz + k] += CPsi_hxy[n]*Psi_hxy[n];
    }
    if (n < total1) {
        int i = n / (nc*nzp1);
        int rem = n - i*(nc*nzp1);
        int ci = rem / nzp1;
        int k = rem - ci*nzp1;
        int j = jBase + ci;
        Hz[i*ny*nzp1 + j*nzp1 + k] += CPsi_hzy[n]*Psi_hzy[n];
    }
}

// Electric: Psi_ezy (nxp1,nc,nz) from Hx; Psi_exy (nx,nc,nzp1) from Hz.
__global__ void k_cpml_y_ele_psi(int nc,int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        int jSrcBase, // Hx/Hz source j for ci=0 (yn:0, yp:n_st-1)
        const float* b_e, const float* a_e,
        const float* Hx, const float* Hz,
        float* Psi_ezy, float* Psi_exy) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int total0 = nxp1*nc*nz;     // Psi_ezy
    int total1 = nx*nc*nzp1;     // Psi_exy
    if (n < total0) {
        int i = n / (nc*nz);
        int rem = n - i*(nc*nz);
        int ci = rem / nz;
        int k = rem - ci*nz;
        int js = jSrcBase + ci;
        float be = b_e[ci], ae = a_e[ci];
        float hx1 = Hx[i*ny*nz + (js+1)*nz + k];
        float hx0 = Hx[i*ny*nz + js*nz + k];
        Psi_ezy[n] = be*Psi_ezy[n] + ae*(hx1-hx0);
    }
    if (n < total1) {
        int i = n / (nc*nzp1);
        int rem = n - i*(nc*nzp1);
        int ci = rem / nzp1;
        int k = rem - ci*nzp1;
        int js = jSrcBase + ci;
        float be = b_e[ci], ae = a_e[ci];
        float hz1 = Hz[i*ny*nzp1 + (js+1)*nzp1 + k];
        float hz0 = Hz[i*ny*nzp1 + js*nzp1 + k];
        Psi_exy[n] = be*Psi_exy[n] + ae*(hz1-hz0);
    }
}

__global__ void k_cpml_y_ele_corr(int nc,int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        int jBase, // Ez/Ex correction j for ci=0 (= e_start)
        float* Ez, float* Ex,
        const float* CPsi_ezy, const float* Psi_ezy,
        const float* CPsi_exy, const float* Psi_exy) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int total0 = nxp1*nc*nz;
    int total1 = nx*nc*nzp1;
    if (n < total0) {
        int i = n / (nc*nz);
        int rem = n - i*(nc*nz);
        int ci = rem / nz;
        int k = rem - ci*nz;
        int j = jBase + ci;
        Ez[i*nyp1*nz + j*nz + k] += CPsi_ezy[n]*Psi_ezy[n];
    }
    if (n < total1) {
        int i = n / (nc*nzp1);
        int rem = n - i*(nc*nzp1);
        int ci = rem / nzp1;
        int k = rem - ci*nzp1;
        int j = jBase + ci;
        Ex[i*nyp1*nzp1 + j*nzp1 + k] += CPsi_exy[n]*Psi_exy[n];
    }
}

// ── Z faces ────────────────────────────────────────────────────────────────
// Layout (ni, nj, nc). Magnetic: Psi_hxz (nxp1,ny,nc) from Ey; Psi_hyz (nx,nyp1,nc) from Ex.
__global__ void k_cpml_z_mag_psi(int nc,int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        int kBase, // 0-based Ey/Ex k-source for ci=0 (= m_start)
        const float* b_m, const float* a_m,
        const float* Ex, const float* Ey,
        float* Psi_hxz, float* Psi_hyz) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int total0 = nxp1*ny*nc;     // Psi_hxz
    int total1 = nx*nyp1*nc;     // Psi_hyz
    if (n < total0) {
        int i = n / (ny*nc);
        int rem = n - i*(ny*nc);
        int j = rem / nc;
        int ci = rem - j*nc;
        int k_h = kBase + ci;
        float bm = b_m[ci], am = a_m[ci];
        float ey1 = Ey[i*ny*nzp1 + j*nzp1 + (k_h+1)];
        float ey0 = Ey[i*ny*nzp1 + j*nzp1 + k_h];
        Psi_hxz[n] = bm*Psi_hxz[n] + am*(ey1-ey0);
    }
    if (n < total1) {
        int i = n / (nyp1*nc);
        int rem = n - i*(nyp1*nc);
        int j = rem / nc;
        int ci = rem - j*nc;
        int k_h = kBase + ci;
        float bm = b_m[ci], am = a_m[ci];
        float ex1 = Ex[i*nyp1*nzp1 + j*nzp1 + (k_h+1)];
        float ex0 = Ex[i*nyp1*nzp1 + j*nzp1 + k_h];
        Psi_hyz[n] = bm*Psi_hyz[n] + am*(ex1-ex0);
    }
}

__global__ void k_cpml_z_mag_corr(int nc,int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        int kBase,
        float* Hx, float* Hy,
        const float* CPsi_hxz, const float* Psi_hxz,
        const float* CPsi_hyz, const float* Psi_hyz) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int total0 = nxp1*ny*nc;
    int total1 = nx*nyp1*nc;
    if (n < total0) {
        int i = n / (ny*nc);
        int rem = n - i*(ny*nc);
        int j = rem / nc;
        int ci = rem - j*nc;
        int k = kBase + ci;
        Hx[i*ny*nz + j*nz + k] += CPsi_hxz[n]*Psi_hxz[n];
    }
    if (n < total1) {
        int i = n / (nyp1*nc);
        int rem = n - i*(nyp1*nc);
        int j = rem / nc;
        int ci = rem - j*nc;
        int k = kBase + ci;
        Hy[i*nyp1*nz + j*nz + k] += CPsi_hyz[n]*Psi_hyz[n];
    }
}

// Electric: Psi_exz (nx,nyp1,nc) from Hy; Psi_eyz (nxp1,ny,nc) from Hx.
__global__ void k_cpml_z_ele_psi(int nc,int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        int kSrcBase, // Hy/Hx k-source for ci=0 (zn:0, zp:n_st-1)
        const float* b_e, const float* a_e,
        const float* Hx, const float* Hy,
        float* Psi_exz, float* Psi_eyz) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int total0 = nx*nyp1*nc;     // Psi_exz
    int total1 = nxp1*ny*nc;     // Psi_eyz
    if (n < total0) {
        int i = n / (nyp1*nc);
        int rem = n - i*(nyp1*nc);
        int j = rem / nc;
        int ci = rem - j*nc;
        int ks = kSrcBase + ci;
        float be = b_e[ci], ae = a_e[ci];
        float hy1 = Hy[i*nyp1*nz + j*nz + (ks+1)];
        float hy0 = Hy[i*nyp1*nz + j*nz + ks];
        Psi_exz[n] = be*Psi_exz[n] + ae*(hy1-hy0);
    }
    if (n < total1) {
        int i = n / (ny*nc);
        int rem = n - i*(ny*nc);
        int j = rem / nc;
        int ci = rem - j*nc;
        int ks = kSrcBase + ci;
        float be = b_e[ci], ae = a_e[ci];
        float hx1 = Hx[i*ny*nz + j*nz + (ks+1)];
        float hx0 = Hx[i*ny*nz + j*nz + ks];
        Psi_eyz[n] = be*Psi_eyz[n] + ae*(hx1-hx0);
    }
}

__global__ void k_cpml_z_ele_corr(int nc,int nx,int ny,int nz,int nxp1,int nyp1,int nzp1,
        int kBase, // Ex/Ey correction k for ci=0 (= e_start)
        float* Ex, float* Ey,
        const float* CPsi_exz, const float* Psi_exz,
        const float* CPsi_eyz, const float* Psi_eyz) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int total0 = nx*nyp1*nc;
    int total1 = nxp1*ny*nc;
    if (n < total0) {
        int i = n / (nyp1*nc);
        int rem = n - i*(nyp1*nc);
        int j = rem / nc;
        int ci = rem - j*nc;
        int k = kBase + ci;
        Ex[i*nyp1*nzp1 + j*nzp1 + k] += CPsi_exz[n]*Psi_exz[n];
    }
    if (n < total1) {
        int i = n / (ny*nc);
        int rem = n - i*(ny*nc);
        int j = rem / nc;
        int ci = rem - j*nc;
        int k = kBase + ci;
        Ey[i*ny*nzp1 + j*nzp1 + k] += CPsi_eyz[n]*Psi_eyz[n];
    }
}

// ════════════════════════ SOURCE INJECTION ════════════════════════════════
// E[idx[n]] += coef[n] * v  where v = voltage_per_e_field[ts].
__global__ void k_inject(int count, const int* idx, const float* coef,
                         float v, float* E) {
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    if (n >= count) return;
    E[idx[n]] += coef[n] * v;
}

// ════════════════════════ SAMPLE REDUCTIONS ═══════════════════════════════
// Block-level partial sums; the partials are summed on the host (counts are
// tiny — a handful of cells). Generic gather+weight reduction.
__global__ void k_gather_sum(int count, const int* idx, const float* w,
                             const float* F, float* partial) {
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    int n = blockIdx.x*blockDim.x + tid;
    float v = (n < count) ? w[n] * F[idx[n]] : 0.0f;
    sh[tid] = v;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] += sh[tid+s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sh[0];
}

// ═══════════════════════════ HOST HELPERS ═════════════════════════════════
static int allocAndCopyF(float** dptr, const float* host, int n) {
    if (n == 0) { *dptr = nullptr; return FDTD_OK; }
    if (cudaMalloc((void**)dptr, (size_t)n*sizeof(float)) != cudaSuccess) {
        std::snprintf(g_lastError, sizeof(g_lastError), "cudaMalloc(%d floats) failed", n);
        return FDTD_ERR;
    }
    if (host) {
        if (cudaMemcpy(*dptr, host, (size_t)n*sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::snprintf(g_lastError, sizeof(g_lastError), "cudaMemcpy H2D failed");
            return FDTD_ERR;
        }
    } else {
        cudaMemset(*dptr, 0, (size_t)n*sizeof(float));
    }
    return FDTD_OK;
}
static int allocAndCopyI(int** dptr, const int* host, int n) {
    if (n == 0) { *dptr = nullptr; return FDTD_OK; }
    if (cudaMalloc((void**)dptr, (size_t)n*sizeof(int)) != cudaSuccess) {
        std::snprintf(g_lastError, sizeof(g_lastError), "cudaMalloc(%d ints) failed", n);
        return FDTD_ERR;
    }
    if (cudaMemcpy(*dptr, host, (size_t)n*sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::snprintf(g_lastError, sizeof(g_lastError), "cudaMemcpy H2D (int) failed");
        return FDTD_ERR;
    }
    return FDTD_OK;
}

// ═══════════════════════════ PUBLIC C API ═════════════════════════════════

int fdtd_device_count(int* count) {
    int c = 0;
    cudaError_t e = cudaGetDeviceCount(&c);
    if (e != cudaSuccess) { *count = 0; return FDTD_ERR; }
    *count = c;
    return FDTD_OK;
}

int fdtd_init(const FdtdDims* d) {
    if (g.inited) fdtd_destroy();
    memset(&g, 0, sizeof(g));
    g.nx=d->nx; g.ny=d->ny; g.nz=d->nz;
    g.nxp1=d->nx+1; g.nyp1=d->ny+1; g.nzp1=d->nz+1;
    g.dx=d->dx; g.dy=d->dy; g.dz=d->dz; g.dt=d->dt;
    g.numberOfTimeSteps = d->numberOfTimeSteps;

    g.nhx = g.nxp1*g.ny*g.nz;
    g.nhy = g.nx*g.nyp1*g.nz;
    g.nhz = g.nx*g.ny*g.nzp1;
    g.nex = g.nx*g.nyp1*g.nzp1;
    g.ney = g.nxp1*g.ny*g.nzp1;
    g.nez = g.nxp1*g.nyp1*g.nz;

    // Fields (zero-initialized)
    if (allocAndCopyF(&g.Hx,nullptr,g.nhx)) return FDTD_ERR;
    if (allocAndCopyF(&g.Hy,nullptr,g.nhy)) return FDTD_ERR;
    if (allocAndCopyF(&g.Hz,nullptr,g.nhz)) return FDTD_ERR;
    if (allocAndCopyF(&g.Ex,nullptr,g.nex)) return FDTD_ERR;
    if (allocAndCopyF(&g.Ey,nullptr,g.ney)) return FDTD_ERR;
    if (allocAndCopyF(&g.Ez,nullptr,g.nez)) return FDTD_ERR;
    g.inited = 1;
    return FDTD_OK;
}

// Upload one coefficient array. `which` enumerates the 18 coeffs.
int fdtd_upload_coeff(int which, const float* host, int n) {
    float** slot = nullptr;
    switch (which) {
        case  0: slot=&g.Cexe;  break; case  1: slot=&g.Cexhz; break; case  2: slot=&g.Cexhy; break;
        case  3: slot=&g.Ceye;  break; case  4: slot=&g.Ceyhx; break; case  5: slot=&g.Ceyhz; break;
        case  6: slot=&g.Ceze;  break; case  7: slot=&g.Cezhy; break; case  8: slot=&g.Cezhx; break;
        case  9: slot=&g.Chxh;  break; case 10: slot=&g.Chxey; break; case 11: slot=&g.Chxez; break;
        case 12: slot=&g.Chyh;  break; case 13: slot=&g.Chyez; break; case 14: slot=&g.Chyex; break;
        case 15: slot=&g.Chzh;  break; case 16: slot=&g.Chzex; break; case 17: slot=&g.Chzey; break;
        default:
            std::snprintf(g_lastError, sizeof(g_lastError), "bad coeff id %d", which);
            return FDTD_ERR;
    }
    return allocAndCopyF(slot, host, n);
}

// Upload a CPML face. The JS driver passes already-flattened f32 arrays in the
// exact layouts cpml.js uses. `faceId`: 0=xn,1=xp,2=yn,3=yp,4=zn,5=zp.
// n0..n3 are element counts for (Psi0/CPsi0 .. Psi3/CPsi3); 0 = unused.
int fdtd_upload_cpml(int faceId, int nc,
        const float* b_e, const float* a_e, const float* b_m, const float* a_m,
        const float* CPsi0, int n0, const float* CPsi1, int n1,
        const float* CPsi2, int n2, const float* CPsi3, int n3,
        int m_start, int e_start, int ascending) {
    if (faceId < 0 || faceId > 5) { std::snprintf(g_lastError,sizeof(g_lastError),"bad face %d",faceId); return FDTD_ERR; }
    CpmlFaceDev& f = g.cpml[faceId];
    f.active = 1; f.nc = nc; f.m_start = m_start; f.e_start = e_start; f.ascending = ascending;
    f.n0=n0; f.n1=n1; f.n2=n2; f.n3=n3;
    if (allocAndCopyF(&f.b_e,b_e,nc)) return FDTD_ERR;
    if (allocAndCopyF(&f.a_e,a_e,nc)) return FDTD_ERR;
    if (allocAndCopyF(&f.b_m,b_m,nc)) return FDTD_ERR;
    if (allocAndCopyF(&f.a_m,a_m,nc)) return FDTD_ERR;
    if (allocAndCopyF(&f.CPsi0,CPsi0,n0)) return FDTD_ERR;
    if (allocAndCopyF(&f.CPsi1,CPsi1,n1)) return FDTD_ERR;
    if (allocAndCopyF(&f.CPsi2,CPsi2,n2)) return FDTD_ERR;
    if (allocAndCopyF(&f.CPsi3,CPsi3,n3)) return FDTD_ERR;
    // Psi arrays are zero-initialized device scratch.
    if (allocAndCopyF(&f.Psi0,nullptr,n0)) return FDTD_ERR;
    if (allocAndCopyF(&f.Psi1,nullptr,n1)) return FDTD_ERR;
    if (allocAndCopyF(&f.Psi2,nullptr,n2)) return FDTD_ERR;
    if (allocAndCopyF(&f.Psi3,nullptr,n3)) return FDTD_ERR;
    return FDTD_OK;
}

int fdtd_upload_vsource(int dir, const int* idx, const float* coef, int count,
                        const float* waveform, int nSteps) {
    g.vsActive = 1; g.vsDir = dir; g.vsCount = count;
    if (allocAndCopyI(&g.vsIdx, idx, count)) return FDTD_ERR;
    if (allocAndCopyF(&g.vsCoef, coef, count)) return FDTD_ERR;
    if (allocAndCopyF(&g.vsWave, waveform, nSteps)) return FDTD_ERR;
    return FDTD_OK;
}

int fdtd_upload_svoltage(int dir, const int* idx, int count, float Csvf) {
    g.svActive = 1; g.svDir = dir; g.svCount = count; g.svCsvf = Csvf;
    if (allocAndCopyI(&g.svIdx, idx, count)) return FDTD_ERR;
    // Unit weight buffer so the generic gather kernel sums raw E values.
    if (cudaMalloc((void**)&g.svUnitW, (size_t)count*sizeof(float)) != cudaSuccess) {
        std::snprintf(g_lastError,sizeof(g_lastError),"svUnitW malloc failed"); return FDTD_ERR;
    }
    {
        float* tmp = (float*)malloc((size_t)count*sizeof(float));
        for (int i=0;i<count;i++) tmp[i]=1.0f;
        cudaMemcpy(g.svUnitW, tmp, (size_t)count*sizeof(float), cudaMemcpyHostToDevice);
        free(tmp);
    }
    int nb = blocks1D(count, 256);
    if (allocAndCopyF(&g.svPartial, nullptr, nb)) return FDTD_ERR;
    cudaMallocHost((void**)&g.svHostPinned, (size_t)nb*sizeof(float));
    return FDTD_OK;
}

// Sampled current is precomputed by the JS driver into 3 weighted index lists
// (one per H component). sum over all three = the loop integral.
int fdtd_upload_scurrent(
        const int* hxIdx, const float* hxW, int hxN,
        const int* hyIdx, const float* hyW, int hyN,
        const int* hzIdx, const float* hzW, int hzN) {
    g.siActive = 1;
    g.siHxN=hxN; g.siHyN=hyN; g.siHzN=hzN;
    if (allocAndCopyI(&g.siHxIdx,hxIdx,hxN)) return FDTD_ERR;
    if (allocAndCopyF(&g.siHxW,hxW,hxN)) return FDTD_ERR;
    if (allocAndCopyI(&g.siHyIdx,hyIdx,hyN)) return FDTD_ERR;
    if (allocAndCopyF(&g.siHyW,hyW,hyN)) return FDTD_ERR;
    if (allocAndCopyI(&g.siHzIdx,hzIdx,hzN)) return FDTD_ERR;
    if (allocAndCopyF(&g.siHzW,hzW,hzN)) return FDTD_ERR;
    int maxN = hxN > hyN ? hxN : hyN; if (hzN > maxN) maxN = hzN;
    int nb = blocks1D(maxN > 0 ? maxN : 1, 256);
    if (allocAndCopyF(&g.siPartial, nullptr, nb)) return FDTD_ERR;
    return FDTD_OK;
}

// ── reduction helper: host-side final sum of device partials ────────────────
static float reduceGather(int count, const int* idx, const float* w,
                          const float* F, float* partial) {
    if (count == 0) return 0.0f;
    int tpb = 256;
    int nb = blocks1D(count, tpb);
    k_gather_sum<<<nb, tpb, tpb*sizeof(float)>>>(count, idx, w, F, partial);
    // Copy partials back and finish on host (nb is small).
    static float hbuf[4096];
    if (nb > 4096) nb = 4096; // safety; sample loops are tiny in practice
    cudaMemcpy(hbuf, partial, (size_t)nb*sizeof(float), cudaMemcpyDeviceToHost);
    float s = 0.0f;
    for (int b = 0; b < nb; b++) s += hbuf[b];
    return s;
}

// ── magnetic CPML dispatch ─────────────────────────────────────────────────
static void runMagneticCPML() {
    const int tpb = 256;
    // X faces (0=xn,1=xp)
    for (int fid = 0; fid <= 1; ++fid) {
        CpmlFaceDev& f = g.cpml[fid];
        if (!f.active) continue;
        int eBase = f.ascending ? f.m_start : 0; // xn: Ez/Ey from i=ci ; xp: from i=n_st+ci
        int hBase = f.m_start;                    // Hy/Hz target i = m_start+ci
        int maxN = (f.n0 > f.n1 ? f.n0 : f.n1);
        int nb = blocks1D(maxN, tpb);
        k_cpml_x_mag_psi<<<nb,tpb>>>(f.nc,g.ny,g.nz,g.nyp1,g.nzp1,eBase,
            f.b_m,f.a_m,g.Ey,g.Ez,f.Psi0,f.Psi1);
        k_cpml_x_mag_corr<<<nb,tpb>>>(f.nc,g.ny,g.nz,g.nyp1,g.nzp1,hBase,
            g.Hy,g.Hz,f.CPsi0,f.Psi0,f.CPsi1,f.Psi1);
    }
    // Y faces (2=yn,3=yp)
    for (int fid = 2; fid <= 3; ++fid) {
        CpmlFaceDev& f = g.cpml[fid];
        if (!f.active) continue;
        int jBase = f.m_start;
        int maxN = (f.n0 > f.n1 ? f.n0 : f.n1);
        int nb = blocks1D(maxN, tpb);
        k_cpml_y_mag_psi<<<nb,tpb>>>(f.nc,g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,jBase,
            f.b_m,f.a_m,g.Hx,g.Hz,f.Psi0,f.Psi1);
        k_cpml_y_mag_corr<<<nb,tpb>>>(f.nc,g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,jBase,
            g.Hx,g.Hz,f.CPsi0,f.Psi0,f.CPsi1,f.Psi1);
    }
    // Z faces (4=zn,5=zp)
    for (int fid = 4; fid <= 5; ++fid) {
        CpmlFaceDev& f = g.cpml[fid];
        if (!f.active) continue;
        int kBase = f.m_start;
        int maxN = (f.n0 > f.n1 ? f.n0 : f.n1);
        int nb = blocks1D(maxN, tpb);
        k_cpml_z_mag_psi<<<nb,tpb>>>(f.nc,g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,kBase,
            f.b_m,f.a_m,g.Ex,g.Ey,f.Psi0,f.Psi1);
        k_cpml_z_mag_corr<<<nb,tpb>>>(f.nc,g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,kBase,
            g.Hx,g.Hy,f.CPsi0,f.Psi0,f.CPsi1,f.Psi1);
    }
}

// ── electric CPML dispatch ─────────────────────────────────────────────────
static void runElectricCPML() {
    const int tpb = 256;
    for (int fid = 0; fid <= 1; ++fid) {
        CpmlFaceDev& f = g.cpml[fid];
        if (!f.active) continue;
        // xn: H source i=ci (base 0); xp: i = n_st+ci-1 = (m_start-1)+ci
        int hBase = f.ascending ? (f.m_start - 1) : 0;
        int eBase = f.e_start;  // Ey/Ez correction i (xn:1, xp:n_st)
        int maxN = (f.n2 > f.n3 ? f.n2 : f.n3);
        int nb = blocks1D(maxN, tpb);
        k_cpml_x_ele_psi<<<nb,tpb>>>(f.nc,g.ny,g.nz,g.nyp1,g.nzp1,hBase,
            f.b_e,f.a_e,g.Hy,g.Hz,f.Psi2,f.Psi3);
        k_cpml_x_ele_corr<<<nb,tpb>>>(f.nc,g.ny,g.nz,g.nyp1,g.nzp1,eBase,
            g.Ey,g.Ez,f.CPsi2,f.Psi2,f.CPsi3,f.Psi3);
    }
    for (int fid = 2; fid <= 3; ++fid) {
        CpmlFaceDev& f = g.cpml[fid];
        if (!f.active) continue;
        int jSrcBase = f.ascending ? (f.m_start - 1) : 0; // yn:0, yp:n_st-1
        int jBase = f.e_start;
        int maxN = (f.n2 > f.n3 ? f.n2 : f.n3);
        int nb = blocks1D(maxN, tpb);
        k_cpml_y_ele_psi<<<nb,tpb>>>(f.nc,g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,jSrcBase,
            f.b_e,f.a_e,g.Hx,g.Hz,f.Psi2,f.Psi3);
        k_cpml_y_ele_corr<<<nb,tpb>>>(f.nc,g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,jBase,
            g.Ez,g.Ex,f.CPsi2,f.Psi2,f.CPsi3,f.Psi3);
    }
    for (int fid = 4; fid <= 5; ++fid) {
        CpmlFaceDev& f = g.cpml[fid];
        if (!f.active) continue;
        int kSrcBase = f.ascending ? (f.m_start - 1) : 0; // zn:0, zp:n_st-1
        int kBase = f.e_start;
        int maxN = (f.n2 > f.n3 ? f.n2 : f.n3);
        int nb = blocks1D(maxN, tpb);
        k_cpml_z_ele_psi<<<nb,tpb>>>(f.nc,g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,kSrcBase,
            f.b_e,f.a_e,g.Hx,g.Hy,f.Psi2,f.Psi3);
        k_cpml_z_ele_corr<<<nb,tpb>>>(f.nc,g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,kBase,
            g.Ex,g.Ey,f.CPsi2,f.Psi2,f.CPsi3,f.Psi3);
    }
}

// Run a batch of [tsStart, tsStart+count) steps. Writes svOut/siOut[ts] for the
// batch (host arrays, length count). Per CONTRACT §2 update order.
int fdtd_run_batch(int tsStart, int count, float* svOut, float* siOut) {
    if (!g.inited) { std::snprintf(g_lastError,sizeof(g_lastError),"not inited"); return FDTD_ERR; }
    dim3 tb(8,8,4);

    for (int s = 0; s < count; ++s) {
        int ts = tsStart + s;

        // 1. updateH (bulk)
        {
            dim3 gHx(blocks1D(g.nz,tb.x), blocks1D(g.ny,tb.y), blocks1D(g.nxp1,tb.z));
            k_updateHx<<<gHx,tb>>>(g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,g.Hx,g.Ey,g.Ez,g.Chxh,g.Chxey,g.Chxez);
            dim3 gHy(blocks1D(g.nz,tb.x), blocks1D(g.nyp1,tb.y), blocks1D(g.nx,tb.z));
            k_updateHy<<<gHy,tb>>>(g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,g.Hy,g.Ez,g.Ex,g.Chyh,g.Chyez,g.Chyex);
            dim3 gHz(blocks1D(g.nzp1,tb.x), blocks1D(g.ny,tb.y), blocks1D(g.nx,tb.z));
            k_updateHz<<<gHz,tb>>>(g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,g.Hz,g.Ex,g.Ey,g.Chzh,g.Chzex,g.Chzey);
        }
        // 2. magnetic CPML
        runMagneticCPML();
        // 3. H captures: sampled current (loop integral) — computed after H update.
        if (g.siActive) {
            float sx = reduceGather(g.siHxN, g.siHxIdx, g.siHxW, g.Hx, g.siPartial);
            float sy = reduceGather(g.siHyN, g.siHyIdx, g.siHyW, g.Hy, g.siPartial);
            float sz = reduceGather(g.siHzN, g.siHzIdx, g.siHzW, g.Hz, g.siPartial);
            if (siOut) siOut[s] = sx + sy + sz;
        }
        // 4. updateE (bulk, interior)
        {
            dim3 gEx(blocks1D(g.nz-1,tb.x), blocks1D(g.ny-1,tb.y), blocks1D(g.nx,tb.z));
            k_updateEx<<<gEx,tb>>>(g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,g.Ex,g.Hz,g.Hy,g.Cexe,g.Cexhz,g.Cexhy);
            dim3 gEy(blocks1D(g.nz-1,tb.x), blocks1D(g.ny,tb.y), blocks1D(g.nx,tb.z));
            k_updateEy<<<gEy,tb>>>(g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,g.Ey,g.Hx,g.Hz,g.Ceye,g.Ceyhx,g.Ceyhz);
            dim3 gEz(blocks1D(g.nz,tb.x), blocks1D(g.ny,tb.y), blocks1D(g.nx,tb.z));
            k_updateEz<<<gEz,tb>>>(g.nx,g.ny,g.nz,g.nxp1,g.nyp1,g.nzp1,g.Ez,g.Hy,g.Hx,g.Ceze,g.Cezhy,g.Cezhx);
        }
        // 5. electric CPML
        runElectricCPML();
        // 6. voltage source injection
        if (g.vsActive) {
            // v = voltage_per_e_field[ts]; fetch from device wave array via host mirror.
            // To avoid a per-step D2H of a scalar we pass v as the device value index.
            // Simpler & exact: read the single waveform scalar back (cheap).
            float v;
            cudaMemcpy(&v, g.vsWave + ts, sizeof(float), cudaMemcpyDeviceToHost);
            float* Etarget = (g.vsDir==0)?g.Ex : (g.vsDir==1)?g.Ey : g.Ez;
            int nb = blocks1D(g.vsCount, 256);
            k_inject<<<nb,256>>>(g.vsCount, g.vsIdx, g.vsCoef, v, Etarget);
        }
        // 7. E captures: sampled voltage
        if (g.svActive) {
            float* Esrc = (g.svDir==0)?g.Ex : (g.svDir==1)?g.Ey : g.Ez;
            // weight is uniform (1.0) here; Csvf applied on host.
            // Reuse gather kernel with unit weights stored implicitly: we pass
            // a null weight? Instead we built svIdx; need a unit-weight buffer.
            // We multiply by Csvf after summing raw field values.
            float raw = reduceGather(g.svCount, g.svIdx, g.svUnitW, Esrc, g.svPartial);
            if (svOut) svOut[s] = g.svCsvf * raw;
        }
        // 8. Far-field DFT accumulation is performed on the HOST by the JS driver
        //    as a documented fallback (see cudaBackend.js). Not done here.
    }
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    return FDTD_OK;
}

// Copy a field array back to host (used by the JS driver to accumulate the
// far-field DFT on the CPU as a documented fallback). `which`: 0=Hx,1=Hy,2=Hz,
// 3=Ex,4=Ey,5=Ez.
int fdtd_read_field(int which, float* host, int n) {
    float* src = nullptr; int len = 0;
    switch (which) {
        case 0: src=g.Hx; len=g.nhx; break;
        case 1: src=g.Hy; len=g.nhy; break;
        case 2: src=g.Hz; len=g.nhz; break;
        case 3: src=g.Ex; len=g.nex; break;
        case 4: src=g.Ey; len=g.ney; break;
        case 5: src=g.Ez; len=g.nez; break;
        default: std::snprintf(g_lastError,sizeof(g_lastError),"bad field id %d",which); return FDTD_ERR;
    }
    if (n != len) { std::snprintf(g_lastError,sizeof(g_lastError),"field %d size mismatch %d!=%d",which,n,len); return FDTD_ERR; }
    CUDA_OK(cudaMemcpy(host, src, (size_t)len*sizeof(float), cudaMemcpyDeviceToHost));
    return FDTD_OK;
}

static void freeF(float** p){ if(*p){ cudaFree(*p); *p=nullptr; } }
static void freeI(int** p){ if(*p){ cudaFree(*p); *p=nullptr; } }

void fdtd_destroy() {
    if (!g.inited) return;
    freeF(&g.Hx);freeF(&g.Hy);freeF(&g.Hz);freeF(&g.Ex);freeF(&g.Ey);freeF(&g.Ez);
    freeF(&g.Cexe);freeF(&g.Cexhz);freeF(&g.Cexhy);
    freeF(&g.Ceye);freeF(&g.Ceyhx);freeF(&g.Ceyhz);
    freeF(&g.Ceze);freeF(&g.Cezhy);freeF(&g.Cezhx);
    freeF(&g.Chxh);freeF(&g.Chxey);freeF(&g.Chxez);
    freeF(&g.Chyh);freeF(&g.Chyez);freeF(&g.Chyex);
    freeF(&g.Chzh);freeF(&g.Chzex);freeF(&g.Chzey);
    for (int i=0;i<6;i++){
        CpmlFaceDev& f=g.cpml[i];
        freeF(&f.b_e);freeF(&f.a_e);freeF(&f.b_m);freeF(&f.a_m);
        freeF(&f.Psi0);freeF(&f.Psi1);freeF(&f.Psi2);freeF(&f.Psi3);
        freeF(&f.CPsi0);freeF(&f.CPsi1);freeF(&f.CPsi2);freeF(&f.CPsi3);
        f.active=0;
    }
    freeI(&g.vsIdx);freeF(&g.vsCoef);freeF(&g.vsWave);
    freeI(&g.svIdx);freeF(&g.svUnitW);freeF(&g.svPartial);
    if(g.svHostPinned){cudaFreeHost(g.svHostPinned);g.svHostPinned=nullptr;}
    freeI(&g.siHxIdx);freeF(&g.siHxW);
    freeI(&g.siHyIdx);freeF(&g.siHyW);
    freeI(&g.siHzIdx);freeF(&g.siHzW);
    freeF(&g.siPartial);
    g.inited=0;
}
