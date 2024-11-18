typedef struct KernelContext_s {
    size_t x;
    size_t maxX;
} KernelContext_t;

__kernel void mxm1D_struct_array(__global KernelContext_t * kc, __global float *a,  __global float *b, __global float *c, const int n) {

	kc[get_global_id(0)].x = get_global_id(0);
	for (int j = 0; j < n; j++) {
		float sum = 0.0;
		for (int k = 0; k < n; k++) {
			sum += a[kc[get_global_id(0)].x * n + k] * b[k * n + j];
		}
		c[kc[get_global_id(0)].x * n + j] = sum;
	}
}

__kernel void mxm1D_struct(__global KernelContext_t * kc, __global float *a,  __global float *b, __global float *c, const int n) {

	kc->x = get_global_id(0);
	for (int j = 0; j < n; j++) {
		float sum = 0.0;
		for (int k = 0; k < n; k++) {
			sum += a[kc->x * n + k] * b[k * n + j];
		}
		c[kc->x * n + j] = sum;
	}
}

__kernel void mxm1D_private(__global KernelContext_t * kc, __global float *a,  __global float *b, __global float *c, const int n) {
	size_t idx = get_global_id(0);
	for (int j = 0; j < n; j++) {
		float sum = 0.0;
		for (int k = 0; k < n; k++) {
			sum += a[idx * n + k] * b[k * n + j];
		}
		c[idx * n + j] = sum;
	}
}
