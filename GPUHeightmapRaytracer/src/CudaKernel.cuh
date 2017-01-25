#pragma once

#include <glm/glm.hpp>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

/*
* Code snippet from
* http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
//#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
//{
//	if (code != cudaSuccess)
//	{
//		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//		if (abort) exit(code);
//	}
//}


/*
 * This class contains all the CUDA related functions and kernels
 *
 * Functions that are callable from the host side are declared with "_w" in the end
 * 
 */
namespace CudaSpace
{
	__host__ void rayTrace_w(dim3 gridSize, dim3 blockSize, unsigned char* buffer, float* d_gpu_pointBuffer);
	__global__ void rayTrace(unsigned char* buffer, float* d_gpu_pointBuffer);
}
