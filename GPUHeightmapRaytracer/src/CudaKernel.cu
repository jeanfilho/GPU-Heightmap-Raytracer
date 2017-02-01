#include "CudaKernel.cuh"
#include <helper_cuda.h>
#include <iostream>
#include <math_functions.hpp>



namespace CudaSpace
{
	
	__device__ int grid_resolution;
	__device__ float *pointBuffer;
	__device__ glm::vec3 *frame_dimensions;
	__device__ glm::vec3 *camera_forward, *camera_right;
	__device__ glm::vec3 *grid_camera_position;
	__device__ glm::ivec2 *texture_resolution;

	/*
	 * GLM::SIGN produces compile error in CUDA
	 */
	__device__ int sign(float input)
	{
		return input >= 0 ? 1 : -1;
	}

	/*
	* Set up parameters for tracing a single ray
	*/
	__device__ void setUpParameters()
	{

	}

	/*
	 *	Cast a ray through the grid
	 *	ray_direction MUST be normalized
	 */
	__device__ glm::uvec3 castRay(glm::vec3& ray_position, glm::vec3& ray_direction, int threadId)
	{
		glm::uvec3 result = glm::zero<glm::uvec3>();
		

		return result;
	}

	/*
	 * Start the ray tracing algorithm for each pixel
	 */
	__global__ void cuda_rayTrace(unsigned char* colorBuffer)
	{
		/*2D Grid and Block*/
		int blockId = blockIdx.x + blockIdx.y * gridDim.x;
		int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

		glm::vec3 ray_position = *grid_camera_position +
			*camera_forward * frame_dimensions->z +
			(threadIdx.x - texture_resolution->y / 2.0f) * glm::cross(*camera_right, *camera_forward) +
			(blockIdx.x - texture_resolution->x / 2.0f) * *camera_right;

		glm::vec3 ray_direction = glm::normalize(ray_position - *grid_camera_position);
		glm::uvec3 color = castRay(ray_position, ray_direction, threadId);
		
		//GL_BGRA
		colorBuffer[threadId * 3] = static_cast<unsigned char>(color.r);
		colorBuffer[threadId * 3 + 1] = static_cast<unsigned char>(color.g);
		colorBuffer[threadId * 3 + 2] = static_cast<unsigned char>(color.b);
	}

	/*
	* Set device parameters
	* TODO: replace with a single Memcpy call?
	*/
	__global__ void cuda_setParameters(int grid_res, glm::ivec2 texture_res, glm::vec3 frame_dim, glm::vec3 camera_for, glm::vec3 camera_rig, float camera_height, float* d_gpu_pointBuffer)
	{
		grid_resolution = grid_res;
		pointBuffer = d_gpu_pointBuffer;
		*frame_dimensions = frame_dim;
		*camera_forward = camera_for;
		*camera_right = camera_rig;
		*grid_camera_position = glm::vec3(grid_resolution / 2.0f, camera_height, grid_resolution / 2.0f);
		*texture_resolution = texture_res;
	}
	__global__ void cuda_initializeDeviceVariables()
	{
		camera_forward = static_cast<glm::vec3*>(malloc(sizeof(glm::vec3)));
		camera_right = static_cast<glm::vec3*>(malloc(sizeof(glm::vec3)));
		grid_camera_position = static_cast<glm::vec3*>(malloc(sizeof(glm::vec3)));
		texture_resolution = static_cast<glm::ivec2*>(malloc(sizeof(glm::ivec2)));
	}
	__global__ void cuda_freeDeviceVariables()
	{
		free(camera_forward);
		free(camera_right);
		free(grid_camera_position);
		free(texture_resolution);
	}

	/*
	 * Set grid and block dimensions, pass parameters to device and call kernels
	 */
	__host__ void rayTrace(glm::ivec2& texture_resolution, glm::vec3& frame_dimensions, glm::vec3& camera_forward, glm::vec3& camera_right, float camera_height, unsigned char* colorBuffer, float* d_gpu_pointBuffer, int gpu_grid_res)
	{
		//TODO: optimize Grid and Block sizes
		dim3 blockSize(texture_resolution.x);
		dim3 gridSize(texture_resolution.y);

		cuda_setParameters << <1, 1 >> > (gpu_grid_res, texture_resolution, frame_dimensions, camera_forward, camera_right, camera_height, d_gpu_pointBuffer);
		checkCudaErrors(cudaDeviceSynchronize());
		cuda_rayTrace << <blockSize, gridSize>> > (colorBuffer);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	__host__ void initializeDeviceVariables()
	{
		cuda_initializeDeviceVariables << <1, 1 >> > ();
		checkCudaErrors(cudaDeviceSynchronize());
	}

	__host__ void freeDeviceVariables()
	{
		cuda_freeDeviceVariables << <1, 1 >> > ();
		checkCudaErrors(cudaDeviceSynchronize());
	}
}
