#include "CudaKernel.cuh"
#include <helper_cuda.h>
#include <iostream>
#include <math_functions.hpp>



namespace CudaSpace
{
	__device__ float frame_distance;
	__device__ int grid_resolution;
	__device__ float *pointBuffer;
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
	* Set up parameters for raytracing
	*/
	__device__ void setUpParameters(const glm::vec3 &ray_direction, float &t_x, float &d_x, int &pos_x, float &t_z, float &d_z, int &pos_z)
	{
		float cell_size = 1;
		// X
		/*If no */
		if (ray_direction.x == 0)
		{
			t_x = FLT_MAX;
			d_x = 0;
		}
		else
		{
			float origin_grid, origin_cell;

			origin_grid = pos_x;
			origin_cell = origin_grid / cell_size;

			if (ray_direction.x > 0)
			{
				d_x = cell_size / ray_direction.x;
				t_x = ((glm::floor(origin_cell) + 1) * cell_size - origin_grid) / ray_direction.x;
			}
			else
			{
				d_x = -cell_size / ray_direction.x;
				t_x = (glm::floor(origin_cell) * cell_size - origin_grid) / ray_direction.x;
			}
		}

		//Z
		if (ray_direction.z == 0)
		{
			t_z = FLT_MAX;
			d_z = 0;
		}
		else
		{
			float origin_grid, origin_cell;

			origin_grid = pos_z;
			origin_cell = origin_grid / cell_size;

			if (ray_direction.z > 0)
			{
				d_z = cell_size / ray_direction.z;
				t_z = ((glm::floor(origin_cell) + 1) * cell_size - origin_grid) / ray_direction.z;
			}
			else
			{
				d_z = -cell_size / ray_direction.z;
				t_z = ((glm::floor(origin_cell)) * cell_size - origin_grid) / ray_direction.z;
			}
		}
	}

	/*
	 *	Cast a ray through the grid
	 *	ray_direction MUST be normalized
	 */
	__device__ glm::uvec3 castRay(glm::vec3& ray_position, glm::vec3& ray_direction, int threadId)
	{
		glm::uvec3 result = glm::zero<glm::uvec3>();
		float t_x, t_z, d_x, d_z;
		int pos_x, pos_z;

		

		pos_x = glm::floor(ray_position.x);
		pos_z = glm::floor(ray_position.z);
		int i = pos_x + pos_z * grid_resolution;
		if (i >= 0 && i < grid_resolution * grid_resolution && pointBuffer[i] >= ray_position.y)
			return result = glm::uvec3(255, 255, 0);
		
		setUpParameters(ray_direction, t_x, d_x, pos_x, t_z, d_z, pos_z);

		while (result == glm::zero<glm::uvec3>())
		{
			if (t_x < t_z)
			{
				ray_position += ray_direction * d_x;
				t_x += d_x;
				if (ray_position.x >= grid_resolution || ray_position.x < 0)
				{
					if (threadId == 153280)
					{
						printf("Out of array at %f %f %f\n", ray_position.x, ray_position.y, ray_position.z);
					}
					return result;
				}
			}
			else
			{
				ray_position += ray_direction * d_z;
				t_z += d_z;
				if (ray_position.x >= grid_resolution || ray_position.x < 0)
				{
					if (threadId == 153280)
					{
						printf("Out of array at %f %f %f\n", ray_position.x, ray_position.y, ray_position.z);
					}
					return result;
				}
			}

			//Check if ray_position.y is inside a range to prevent the thread being stuck on a loop
			if (ray_position.y < 0 || ray_position.y >= 50)
				return result;

			pos_x = glm::floor(ray_position.x);
			pos_z = glm::floor(ray_position.z);
			i = pos_x + pos_z *grid_resolution;
			if (i >= 0 && i < grid_resolution * grid_resolution && pointBuffer[i] >= ray_position.y)
			{
				float reduc = ( (pointBuffer[i] < 0 ? -pointBuffer[i] : pointBuffer[i] / 50.0f));
				unsigned char red = 255;
				result = glm::uvec3(red, 0, 0);
			}
		}

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
			*camera_forward * frame_distance +
			(threadIdx.x - (*texture_resolution).y / 2.0f) * glm::cross(*camera_right, *camera_forward) +
			(blockIdx.x - (*texture_resolution).x / 2.0f) * *camera_right;

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
	__global__ void cuda_setParameters(int grid_res, glm::ivec2 texture_res, float frame_dis, glm::vec3 camera_for, glm::vec3 camera_rig, float camera_height, float* d_gpu_pointBuffer)
	{
		grid_resolution = grid_res;
		frame_distance = frame_dis;
		pointBuffer = d_gpu_pointBuffer;
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
	__host__ void rayTrace(glm::ivec2& texture_resolution, float frame_distance, glm::vec3& camera_forward, glm::vec3& camera_right, float camera_height, unsigned char* colorBuffer, float* d_gpu_pointBuffer, int gpu_grid_res)
	{
		//TODO: optimize Grid and Block sizes
		dim3 blockSize(texture_resolution.x);
		dim3 gridSize(texture_resolution.y);

		cuda_setParameters << <1, 1 >> > (gpu_grid_res, texture_resolution, frame_distance, camera_forward, camera_right, camera_height, d_gpu_pointBuffer);
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
