// fdtd_cuda.h — C interface between the CUDA solver (fdtd_cuda.cu) and the
// Node-API glue (binding.cpp). Plain C linkage so the host compiler and nvcc
// agree on symbol names.

#ifndef FDTD_CUDA_H
#define FDTD_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

#define FDTD_OK  0
#define FDTD_ERR 1

typedef struct {
    int nx, ny, nz;
    float dx, dy, dz, dt;
    int numberOfTimeSteps;
} FdtdDims;

// Returns last error string set by any failing call (never null).
const char* fdtd_last_error();

// Number of CUDA-capable devices (0 if none / driver missing).
int fdtd_device_count(int* count);

// Allocate device fields for a problem of the given dimensions.
int fdtd_init(const FdtdDims* d);

// Upload one of the 18 coefficient arrays (see .cu for the id mapping).
int fdtd_upload_coeff(int which, const float* host, int n);

// Upload a CPML face. Layouts must match cpml.js. faceId: 0=xn..5=zp.
// Psi0/CPsi0 = first  magnetic array, Psi1/CPsi1 = second magnetic array,
// Psi2/CPsi2 = first  electric array, Psi3/CPsi3 = second electric array.
int fdtd_upload_cpml(int faceId, int nc,
        const float* b_e, const float* a_e, const float* b_m, const float* a_m,
        const float* CPsi0, int n0, const float* CPsi1, int n1,
        const float* CPsi2, int n2, const float* CPsi3, int n3,
        int m_start, int e_start, int ascending);

// Single voltage source. dir: 0=x,1=y,2=z.
int fdtd_upload_vsource(int dir, const int* idx, const float* coef, int count,
                        const float* waveform, int nSteps);

// Single sampled voltage observer. dir: 0=x,1=y,2=z.
int fdtd_upload_svoltage(int dir, const int* idx, int count, float Csvf);

// Sampled current observer expressed as 3 weighted gather lists (one per H comp).
int fdtd_upload_scurrent(
        const int* hxIdx, const float* hxW, int hxN,
        const int* hyIdx, const float* hyW, int hyN,
        const int* hzIdx, const float* hzW, int hzN);

// Run steps [tsStart, tsStart+count). svOut/siOut are host arrays length=count
// receiving the per-step sampled voltage / current scalars (may be null).
int fdtd_run_batch(int tsStart, int count, float* svOut, float* siOut);

// Copy a resident field back to host. which: 0=Hx..5=Ez.
int fdtd_read_field(int which, float* host, int n);

// Free all device resources.
void fdtd_destroy();

#ifdef __cplusplus
}
#endif

#endif // FDTD_CUDA_H
