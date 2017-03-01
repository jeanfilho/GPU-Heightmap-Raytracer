#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/detail/func_common.hpp>
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
	__global__ struct Color
	{
		__device__ __host__ Color(){ r = 0; g = 0; b = 0; }
		__device__ __host__ Color(unsigned char r, unsigned char g, unsigned char b) { this->r = r; this->g = g; this->b = b; }
		__device__ __host__ Color(unsigned short r, unsigned short g, unsigned short b)
		{
			this->r = unsigned char (glm::floor(r / static_cast<float>(USHRT_MAX) * 255.f));
			this->g = unsigned char (glm::floor(g / static_cast<float>(USHRT_MAX) * 255.f));
			this->b = unsigned char (glm::floor(b / static_cast<float>(USHRT_MAX) * 255.f));
		}
		unsigned char r, g, b;
	};
	__host__ void rayTrace(glm::ivec2& texture_resolution, glm::vec3& frame_dimensions, glm::vec3& camera_forward, glm::vec3& grid_camera_position, unsigned char* colorBuffer, bool use_color, float max_height);
	__host__ void initializeDeviceVariables(glm::ivec2& point_buffer_res, glm::ivec2& texture_res, float* d_gpu_pointBuffer, CudaSpace::Color* d_color_map, int LOD_levels, int stride_x, float max_height);
	__host__ void freeDeviceVariables();
}
