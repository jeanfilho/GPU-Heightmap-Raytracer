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
	__host__ void rayTrace(glm::ivec2& texture_resolution, glm::vec3& frame_dimensions, glm::vec3& camera_forward, float camera_height, unsigned char* colorBuffer);
	__host__ void initializeDeviceVariables(int grid_res, glm::ivec2& texture_res, float* d_gpu_pointBuffer, unsigned char* d_color_map, glm::ivec2& color_map_res);
	__host__ void freeDeviceVariables();
}
