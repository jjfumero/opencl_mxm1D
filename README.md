## Matrix Multiplication in OpenCL

This program shows different 1D kernels for the Matrix Multiplication.

### How to compile?

```bash
make
```

### How to run? 

```bash
./mxm
```

### Options:

```bash
./mxm -h
Options: 
	 -p <number>       Select an OpenCL Platform Number
	 -s <size>         Select input matrix size
	 -k <kernel name>  Input Kernel < mxm1D_private | mxm1D_struct_array | mxm1D_struct >
	 -w <nThreads>     Select local work group size <nThreads x nThreads>. If not selected, then it sets to NULL
	 -f                Apply optimizations in the compiler flags when building the kernel (-cl-mad-enable -cl-fast-relaxed-math -w)
	 -c                Check results
	 -h                Show this help
```

#### Examples:

Run on platform `1` with size `1024x1024`, checking results `on` and kernel `mxm1D_private`:

```bash
./mxm -c -p 1 -s 1024 -k mxm1D_private
```


Running array of struct version to store the thread-id, checking results `on` and platform `1` with size `1024x1024` 

```bash
./mxm -c -p 1 -s 1024 -k mxm1D_struct_array
```


Running with the struct version to store the thread-id, checking results `on` and platform `1` with size `1024x1024` 

```bash
./mxm -c -p 1 -s 1024 -k mxm1D_struct
```
