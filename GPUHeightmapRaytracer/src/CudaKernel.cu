#include "CudaKernel.cuh"

#define GRID1D_BLOCK1D

namespace CudaSpace
{
	__host__ void rayTrace_w(dim3 gridSize, dim3 blockSize, unsigned char* buffer, float* d_gpu_pointBuffer)
	{
		rayTrace << <gridSize, blockSize >> > (buffer, d_gpu_pointBuffer);
	}

	__global__ void rayTrace(unsigned char* buffer, float* d_gpu_pointBuffer)
	{
#ifdef GRID1D_BLOCK1D
		/*1D Grid and BLock*/
		int blockId = blockIdx.x;
		int threadId = blockId * (blockDim.x) + threadIdx.x;
#else
		/*2D Grid and Block*/
		int blockId = blockIdx.x + blockIdx.y * gridDim.x;
		int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
#endif
		//TODO: implemented proper raytracing
		buffer[threadId] = 255;
		buffer[threadId + 1] = 0;
		buffer[threadId + 2] = 0;
		buffer[threadId + 3] = 255;
	}

}
