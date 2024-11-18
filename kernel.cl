// Other kernels to experiments with generated kernels from OpenJDK Babylon
// (HAT).
typedef struct KernelContext_s {
  size_t x;   
  size_t maxX;  
} KernelContext_t;

// Initial OpenCL version to perform a naive Matrix Multiplication in 1D.
// This is just for testing.
__kernel void mxm1D_private(__global KernelContext_t *kc, __global float *a,
                            __global float *b, __global float *c, const int n) {
  size_t idx = get_global_id(0);  // 
  for (int j = 0; j < n; j++) {
    float sum = 0.0;
    for (int k = 0; k < n; k++) {
      sum += a[idx * n + k] * b[k * n + j];
    }
    c[idx * n + j] = sum;
  }
}

// This version uses an array of struct to access the thread-id. This is more
// expensive that the previous kernel due to access to global memory.
// [IMPORTANT] This kernel is just for testing.
__kernel void mxm1D_struct_array(__global KernelContext_t *kc,
                                 __global float *a, __global float *b,
                                 __global float *c, const int n) {
  kc[get_global_id(0)].x = get_global_id(0);
  for (int j = 0; j < n; j++) {
    float sum = 0.0;
    for (int k = 0; k < n; k++) {
      sum += a[kc[get_global_id(0)].x * n + k] * b[k * n + j];
    }
    c[kc[get_global_id(0)].x * n + j] = sum;
  }
}

// This version produces WRONG RESULTS due to a race condition. This is just for
// testing and comparing the kernel time with other versions, since OpenJDK
// Babylon, as in November 2024, produces a similar kernel. [IMPORTANT] This
// kernel is just for testing.
__kernel void mxm1D_struct(__global KernelContext_t *kc, __global float *a,
                           __global float *b, __global float *c, const int n) {

  kc->x = get_global_id(0);
  for (int j = 0; j < n; j++) {
    float sum = 0.0;
    for (int k = 0; k < n; k++) {
      sum += a[kc->x * n + k] * b[k * n + j];
    }
    c[kc->x * n + j] = sum;
  }
}
