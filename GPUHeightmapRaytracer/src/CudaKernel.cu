#include "CudaKernel.cuh"
#include <helper_cuda.h>
#include <iostream>
#include <math_functions.hpp>



namespace CudaSpace
{
	
	__device__ int grid_resolution;
	__device__ float minimun_ray_inclination = 0.01f;
	__device__ float cell_size[] = {1,2,4};
	__device__ float *point_buffer;
	__device__ float maxHeight, minHeight;
	__device__ glm::vec3 *frame_dimension;
	__device__ glm::vec3 *camera_forward;
	__device__ glm::vec3 *grid_camera_position;
	__device__ glm::ivec2 *texture_resolution;
	__device__ glm::mat3x3 *pixel_to_grid_matrix;

	/*
	 * GLM::SIGN produces compile error in CUDA
	 */
	__device__ int sign(float input)
	{
		return input >= 0 ? 1 : -1;
	}

	/*
	* Set up parameters for tracing a single ray
	* 
	* Ray always starts in the grid
	*/
	__device__ void setUpParameters(float &tMax, float &tDelta, char &step, int &pos, float ray_origin, float ray_direction, int LOD)
	{
		/*Set starting voxel*/
		pos = static_cast<int>(glm::floor(ray_origin));

		/*Set whether position should increment or decrement based on ray direction*/
		step = sign(ray_direction);

		/*
		 *Calculate the first intersection in the grid and
		 *the distance between each axis intersection with the grid
		 */
		if (ray_direction == 0)
		{
			tMax = FLT_MAX;
			tDelta = 0;
		}
		else
		{
			tMax = ((glm::floor(ray_origin / cell_size[LOD]) + (ray_direction < 0 ? 0 : 1)) * cell_size[LOD] - ray_origin) / ray_direction;
			tDelta = (ray_direction < 0 ? -1 : 1) * cell_size[LOD] / ray_direction;
		}
	}


	/*
	 *	Cast a ray through the grid {Amanatides, 1987 #22}
	 *	ray_direction MUST be normalized
	 */
	__device__ glm::uvec3 castRay(glm::vec3& ray_origin, glm::vec3& ray_direction)
	{
		float tMaxX, tMaxZ, tDeltaX, tDeltaZ;
		int posX, posZ;
		char stepX, stepZ;

		int index = 0;
		float tMax = 0;
		glm::vec3 posVector;

		setUpParameters(tMaxX, tDeltaX, stepX, posX, ray_origin.x, ray_direction.x, 0);
		setUpParameters(tMaxZ, tDeltaZ, stepZ, posZ, ray_origin.z, ray_direction.z, 0);

		/*Check if ray starts outside of the grid*/
		if (posX >= grid_resolution || posX < 0 ||
			posZ >= grid_resolution || posZ < 0)
			return glm::zero<glm::uvec3>();

		/*Check if ray starts under the heightmap*/
		index = posX * grid_resolution + posZ;
		if (point_buffer[index] >= ray_origin.y)
			return  glm::zero<glm::uvec3>();


		/*Loop the DDA algorithm*/
		while(true)
		{

			/*Advance ray through the grid*/
			if(tMaxX < tMaxZ)
			{
				posVector = ray_origin + ray_direction * tMax;
				tMaxX += tDeltaX;
				tMax = tMaxX;
				posX += stepX;
			}
			else
			{
				posVector = ray_origin + ray_direction * tMax;
				tMaxZ += tDeltaZ;
				tMax = tMaxZ;
				posZ += stepZ;
			}
			/*Check if ray is outside of the grid*/
			if (posX >= grid_resolution || posX < 0 ||
				posZ >= grid_resolution || posZ < 0)
			{
				return glm::zero<glm::uvec3>();
			}

			/*Check if ray intersects*/
			index = posX * grid_resolution + posZ;
			if (point_buffer[index] >= posVector.y)
			{
				float factor = point_buffer[index] / (maxHeight - minHeight);
				return glm::uvec3(255*factor, 0, 0);
			}
		}
	}
	
	/*
	 * Converts a pixel position to the grid space
	 * Pinhole camera model - From: Realistic Ray Tracing by Peter Shirley, pages 37-42
	 */
	__device__ glm::vec3 pixelToWindowSpace(glm::ivec2 &pixel_position)
	{
		glm::vec3 result = glm::vec3(
			-frame_dimension->x / 2.0f + (frame_dimension->x) * pixel_position.x / (texture_resolution->x - 1),
			-frame_dimension->y / 2.0f + (frame_dimension->y) * pixel_position.y / (texture_resolution->y - 1),
			-frame_dimension->z);
		return result;
	}


	/*
	 * Start the ray tracing algorithm for each pixel
	 */
	__global__ void cuda_rayTrace(unsigned char* colorBuffer)
	{
		/*2D Grid and Block*/
		/*int blockId = blockIdx.x + blockIdx.y * gridDim.x;
		int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;*/

		/*1D Grid and Block*/
		/*int threadId = blockIdx.x *blockDim.x + threadIdx.x;*/

		/*2D Grid and 1D Block*/
		int threadId = blockIdx.y * gridDim.x + blockIdx.x;

		glm::uvec3 color;
		glm::vec3 ray_direction;
		glm::ivec2 pixel_position;

		/*Get the pixel position of this thread*/
		pixel_position = glm::ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

		/*Calculate ray direction and cast ray*/
		ray_direction = glm::normalize(*pixel_to_grid_matrix * pixelToWindowSpace(pixel_position));
		color = castRay(*grid_camera_position, ray_direction);

		//GL_BGRA
		colorBuffer[threadId * 3] = static_cast<unsigned char>(color.r);
		colorBuffer[threadId * 3 + 1] = static_cast<unsigned char>(color.g);
		colorBuffer[threadId * 3 + 2] = static_cast<unsigned char>(color.b);
	}

	/*
	* Set device parameters
	* TODO: replace with a single Memcpy call?
	*/
	__global__ void cuda_setParameters(int grid_res, glm::ivec2 texture_res, glm::vec3 frame_dim, glm::vec3 camera_for, float camera_height, float* d_gpu_pointBuffer, float minH, float maxH)
	{
		grid_resolution = grid_res;
		point_buffer = d_gpu_pointBuffer;
		minHeight = minH;
		maxHeight = maxH;
		*frame_dimension = frame_dim;
		*grid_camera_position = glm::vec3(grid_resolution / 2.0f, camera_height, grid_resolution / 2.0f);
		*texture_resolution = texture_res;

		/*Basis change matrix - changing from left to right handed*/
		glm::vec3 u, v, w;
		w = camera_for;
		u = glm::normalize(glm::cross(glm::vec3(0, 1, 0), w));
		v = glm::cross(w, u);
		*pixel_to_grid_matrix =
			glm::mat3x3(
				 u.x, v.x, w.x,
				 u.y, v.y, w.y,
				-u.z,-v.z,-w.z
			);

	}
	__global__ void cuda_initializeDeviceVariables()
	{
		frame_dimension = static_cast<glm::vec3*>(malloc(sizeof(glm::vec3)));
		grid_camera_position = static_cast<glm::vec3*>(malloc(sizeof(glm::vec3)));
		texture_resolution = static_cast<glm::ivec2*>(malloc(sizeof(glm::ivec2)));
		pixel_to_grid_matrix = static_cast<glm::mat3x3*>(malloc(sizeof(glm::mat3x3)));
	}
	__global__ void cuda_freeDeviceVariables()
	{
		free(grid_camera_position);
		free(texture_resolution);
		free(frame_dimension);
		free(pixel_to_grid_matrix);
	}

	/*
	 * Set grid and block dimensions, pass parameters to device and call kernels
	 */
	__host__ void rayTrace(glm::ivec2& texture_resolution, glm::vec3& frame_dimensions, glm::vec3& camera_forward, float camera_height, unsigned char* colorBuffer, float* d_gpu_pointBuffer, int gpu_grid_res, float minHeight, float maxHeight)
	{
		//TODO: optimize Grid and Block sizes
		dim3 gridSize(texture_resolution.x, texture_resolution.y);
		dim3 blockSize(1);

		cuda_setParameters << <1, 1 >> > (gpu_grid_res, texture_resolution, frame_dimensions, camera_forward, camera_height, d_gpu_pointBuffer, minHeight, maxHeight);
		checkCudaErrors(cudaDeviceSynchronize());
		cuda_rayTrace << <gridSize, blockSize >> > (colorBuffer);
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
