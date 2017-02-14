#include "CudaKernel.cuh"
#include <helper_cuda.h>
#include <iostream>
#include <math_functions.hpp>



namespace CudaSpace
{
	
	__device__ float minimum_ray_inclination = 0.01f;
	__device__ float *point_buffer;
	__device__ unsigned char *color_map;
	__device__ glm::vec2 *cell_size;
	__device__ glm::vec3 *frame_dimension;
	__device__ glm::vec3 *camera_forward;
	__device__ glm::vec3 *grid_camera_position;
	__device__ glm::ivec2 *texture_resolution;
	__device__ glm::ivec2 *point_buffer_resolution;
	__device__ glm::ivec2 *color_map_resolution;
	__device__ glm::mat3x3 *pixel_to_grid_matrix;

	/*Get a colormap index from a height map index*/
	__device__ int getColorMapIndex(int posX, int posZ)
	{
		int colorX, colorZ;
		colorX = float(posX) / point_buffer_resolution->x * color_map_resolution->x + 50;
		colorZ = float(posZ) / point_buffer_resolution->y * color_map_resolution->y + 50;
		return colorX + colorZ * color_map_resolution->x;
	}

	/*
	 * GLM::SIGN produces compile error in CUDA
	 */
	__device__ char sign(float input)
	{
		return input >= 0 ? 1 : -1;
	}

	/*
	* Set up parameters for tracing a single ray
	* 
	* Ray always starts in the grid
	*/
	__device__ void setUpParameters(float &tMax, float &tDelta, char &step, int &pos, float ray_origin, float ray_direction, int LOD, float cell_dimension)
	{
		/*Set starting voxel*/
		pos = static_cast<int>(glm::floor(ray_origin));

		/*Set whether position should increment or decrement based on ray direction*/
		step = sign(ray_direction);

		/*
		 *Calculate the first intersection in the grid and
		 *the distance between each axis intersection with the grid
		 */
		if (glm::abs(ray_direction) < 0.00001f)
		{
			tMax = FLT_MAX;
			tDelta = 0;
		}
		else
		{
			tMax = ((glm::floor(ray_origin / cell_dimension) + (ray_direction < 0 ? 0 : 1)) * cell_dimension - ray_origin) / ray_direction;
			tDelta = (ray_direction < 0 ? -1 : 1) * cell_dimension / ray_direction;
		}
	}

	/*
	 *	Cast a ray through the grid {Amanatides, 1987 #22}
	 *	ray_direction MUST be normalized
	 */
	__device__ int castRay(glm::vec3& ray_origin, glm::vec3& ray_direction)
	{
		float tMaxX, tMaxZ, tDeltaX, tDeltaZ;
		int posX, posZ;
		char stepX, stepZ;

		int height_index;
		glm::dvec3 current_ray_position;

		setUpParameters(tMaxX, tDeltaX, stepX, posX, ray_origin.x, ray_direction.x, 0, cell_size->x);
		setUpParameters(tMaxZ, tDeltaZ, stepZ, posZ, ray_origin.z, ray_direction.z, 0, cell_size->y);

		/*Check if ray starts outside of the grid*/
		if (posX >= point_buffer_resolution->x || posX < 0 || posZ >= point_buffer_resolution->y || posZ < 0)
			return -1;

		/*Check if ray starts under the heightmap*/
		height_index = posX + point_buffer_resolution->x * posZ;
		if (point_buffer[height_index] >= ray_origin.y)
			return  -1;

		/*Loop the DDA algorithm*/
		while(true)
		{
			/*Check if ray intersects - Texel intersection from Dick, C., et al. (2009). GPU ray-casting for scalable terrain rendering. Proceedings of EUROGRAPHICS, Citeseer. Page */
			if (ray_direction.y >= 0)
				current_ray_position = ray_origin + ray_direction * (tMaxX < tMaxZ ? tDeltaX + tMaxX : tDeltaZ + tMaxZ);
			else
				current_ray_position = ray_origin + ray_direction * glm::min(tMaxX, tMaxZ);
						
			height_index = posX + point_buffer_resolution->x * posZ;
			if (point_buffer[height_index] >= current_ray_position.y)
			{
				return getColorMapIndex(posX, posZ) * 3;
			}

			/*Advance ray through the grid*/
			if(tMaxX < tMaxZ)
			{
				tMaxX += tDeltaX;
				posX += stepX;
			}
			else
			{
				tMaxZ += tDeltaZ;
				posZ += stepZ;
			}

			/*Check if ray is outside of the grid*/
			if (posX >= point_buffer_resolution->x || posX < 0 || posZ >= point_buffer_resolution->y || posZ < 0 || current_ray_position.y < 0)
			{
				return -1;
			}
		}
	}
	
	/*
	 * Converts a pixel position to the grid space
	 * Pinhole camera model - From: Realistic Ray Tracing by Peter Shirley, pages 37-42
	 */
	__device__ glm::vec3 viewToGridSpace(glm::ivec2 &pixel_position)
	{
		glm::vec3 result = glm::vec3(
			frame_dimension->x / 2.0f - (frame_dimension->x) * pixel_position.x / (texture_resolution->x - 1),
			-frame_dimension->y / 2.0f + (frame_dimension->y) * pixel_position.y / (texture_resolution->y - 1),
			-frame_dimension->z);
		return result;
	}


	/*
	 * Start the ray tracing algorithm for each pixel
	 */
	__global__ void cuda_rayTrace(unsigned char* color_buffer)
	{
		/*2D Grid and Block*/
		/*int blockId = blockIdx.x + blockIdx.y * gridDim.x;
		int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;*/

		/*1D Grid and Block*/
		/*int threadId = blockIdx.x *blockDim.x + threadIdx.x;*/

		/*2D Grid and 1D Block*/
		int threadId = blockIdx.y * gridDim.x + blockIdx.x;

		int color_index;
		glm::vec3 ray_direction, ray_position;
		glm::ivec2 pixel_position;

		/*Get the pixel position of this thread*/
		pixel_position = glm::ivec2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

		/*Calculate ray direction and cast ray*/
		ray_direction = (*pixel_to_grid_matrix * viewToGridSpace(pixel_position));

		ray_position = ray_direction + *grid_camera_position;
		ray_direction = glm::normalize(ray_direction);
		color_index = castRay(ray_position, ray_direction);
		
		//GL_BGRA
		if(color_index >= 0)
		{
			color_buffer[threadId * 3] = color_map[color_index];
			color_buffer[threadId * 3 + 1] = color_map[color_index + 1];
			color_buffer[threadId * 3 + 2] = color_map[color_index + 2];
		}
		else
		{
			color_buffer[threadId * 3] = 200;
			color_buffer[threadId * 3 + 1] = 200;
			color_buffer[threadId * 3 + 2] = 200;
		}
	}

	/*
	* Set device parameters
	* TODO: replace with a single Memcpy call?
	*/
	__global__ void cuda_setParameters(glm::vec3 frame_dim, glm::vec3 camera_for, float camera_height)
	{
		*frame_dimension = frame_dim;
		*grid_camera_position = glm::vec3(point_buffer_resolution->x / 2.0f, camera_height, point_buffer_resolution->y / 3.0f);

		/*Basis change matrix from view to grid space*/
		glm::vec3 u, v, w;
		w = -camera_for;
		u = glm::normalize(glm::cross(glm::vec3(0, 100, 0), w));
		v = glm::cross(w, u);
		*pixel_to_grid_matrix = (glm::mat3x3(u,v,w));

	}
	__global__ void cuda_initializeDeviceVariables(glm::ivec2 point_buffer_res, glm::ivec2 texture_res, float* d_gpu_pointBuffer, unsigned char *d_color_map, glm::ivec2 color_map_res, glm::vec2 cell_siz)
	{
		frame_dimension = new glm::vec3();
		grid_camera_position = new glm::vec3();
		texture_resolution = new glm::ivec2();
		color_map_resolution = new glm::ivec2();
		point_buffer_resolution = new glm::ivec2();
		pixel_to_grid_matrix = new glm::mat3x3();
		cell_size = new glm::vec2();

		*point_buffer_resolution = point_buffer_res;
		*texture_resolution = texture_res;	
		point_buffer = d_gpu_pointBuffer;
		color_map = d_color_map;
		*color_map_resolution = color_map_res;
		*cell_size = cell_siz;
	}
	__global__ void cuda_freeDeviceVariables()
	{
		delete(grid_camera_position);
		delete(texture_resolution);
		delete(frame_dimension);
		delete(pixel_to_grid_matrix);
		delete(color_map_resolution);
		delete(point_buffer_resolution);
		delete(cell_size);
	}

	/*
	 * Set grid and block dimensions, pass parameters to device and call kernels
	 */
	__host__ void rayTrace(glm::ivec2& texture_resolution, glm::vec3& frame_dimensions, glm::vec3& camera_forward, float camera_height, unsigned char* color_buffer)
	{
		//TODO: optimize Grid and Block sizes
		dim3 gridSize(texture_resolution.x, texture_resolution.y);
		dim3 blockSize(1);

		cuda_setParameters << <1, 1 >> > (frame_dimensions, camera_forward, camera_height);
		checkCudaErrors(cudaDeviceSynchronize());
		cuda_rayTrace << <gridSize, blockSize >> > (color_buffer);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	__host__ void initializeDeviceVariables(glm::ivec2& point_buffer_res, glm::ivec2& texture_res, float* d_gpu_pointBuffer, unsigned char* d_color_map, glm::ivec2& color_map_res, glm::vec2& cell_size)
	{
		cuda_initializeDeviceVariables << <1, 1 >> > (point_buffer_res, texture_res, d_gpu_pointBuffer, d_color_map, color_map_res, cell_size);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	__host__ void freeDeviceVariables()
	{
		cuda_freeDeviceVariables << <1, 1 >> > ();
		checkCudaErrors(cudaDeviceSynchronize());
	}
}
