#include "CudaKernel.cuh"
#include <helper_cuda.h>
#include <iostream>
#include <math_functions.hpp>

namespace CudaSpace
{
	__device__ float max_height = 0;
	__device__ float *point_buffer;
	__device__ unsigned char *color_map;
	__device__ int LOD_MAX = 2;
	__device__ int LOD_jump_cells[] = {1, 2, 4};
	__device__ float LOD_distance[] = {0, 1000, 2000};
	__device__ glm::vec2 *cell_size;
	__device__ glm::vec3 *frame_dimension;
	__device__ glm::vec3 *camera_forward;
	__device__ glm::vec3 *grid_camera_position;
	__device__ glm::ivec2 *texture_resolution;
	__device__ glm::ivec2 *point_buffer_resolution;
	__device__ glm::ivec2 *color_map_resolution;
	__device__ glm::mat3x3 *pixel_to_grid_matrix;

	/*
	 *Get a colormap index from a height map index
	 */
	__device__ void getColorMapIndex(int posX, int posZ, glm::uvec3& result)
	{
		int colorX, colorZ;
		colorX = float(posX) / float(point_buffer_resolution->x) * color_map_resolution->x;
		colorZ = float(posZ) / float(point_buffer_resolution->y) * color_map_resolution->y;
		int index = (colorX + colorZ * color_map_resolution->x) * 3;
		result = glm::uvec3(color_map[index], color_map[index + 1], color_map[index + 2]);
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
	 */
	__device__ void setUpParameters(float &tMax, float &tDelta, char &step, int &pos, float ray_origin, float ray_direction, float cell_dimension)
	{
		/*Set starting voxel*/
		pos = static_cast<int>(floor(ray_origin / cell_dimension));

		/*Set whether position should increment or decrement based on ray direction*/
		step = sign(ray_direction);

		/*
		 *Calculate the first intersection in the grid and
		 *the distance between each axis intersection with the grid
		 */
		if (abs(ray_direction) < 0.00001f)
		{
			tMax = FLT_MAX;
			tDelta = 0;
		}
		else
		{
			tMax = ((floor(ray_origin / cell_dimension) + (ray_direction < 0 ? 0 : 1)) * cell_dimension - ray_origin) / ray_direction;
			tDelta = (ray_direction < 0 ? -1 : 1) * cell_dimension / ray_direction;
		}
	}

	/*
	 *	Cast a ray through the grid {Amanatides, 1987 #22}
	 *	ray_direction MUST be normalized
	 */
	__device__ void castRay(glm::vec3& ray_origin, glm::vec3& ray_direction, glm::uvec3& result)
	{
		float tMaxX, tMaxZ, tDeltaX, tDeltaZ, height_value;
		int posX, posZ, height_index, LOD = 1;
		char stepX, stepZ;
		glm::vec3 current_ray_position;

		setUpParameters(tMaxX, tDeltaX, stepX, posX, ray_origin.x, ray_direction.x, cell_size->x);
		setUpParameters(tMaxZ, tDeltaZ, stepZ, posZ, ray_origin.z, ray_direction.z, cell_size->y);

		/*Check if ray starts outside of the grid*/
		if (posX >= point_buffer_resolution->x || posX < 0 || posZ >= point_buffer_resolution->y || posZ < 0)
			return;

		/*Check if ray starts under the heightmap*/
		height_index = posX + point_buffer_resolution->x * posZ;
		if (point_buffer[height_index] >= ray_origin.y)
			return;

		/*Loop the DDA algorithm*/
		while(true)
		{
			/*Check if ray intersects - Texel intersection from Dick, C., et al. (2009). GPU ray-casting for scalable terrain rendering. Proceedings of EUROGRAPHICS, Citeseer. Page */
			if (ray_direction.y >= 0)
				current_ray_position = ray_origin + ray_direction * (tMaxX < tMaxZ ? tDeltaX + tMaxX : tDeltaZ + tMaxZ);
			else
				current_ray_position = ray_origin + ray_direction * glm::min(tMaxX, tMaxZ);
						
			height_index = posX + point_buffer_resolution->x * posZ;
			height_value = 0;
			for(int i = 0; i < LOD; i++)
			{
				for (int j = 0; j < LOD; j++)
				{
					height_value += point_buffer[height_index - stepX * i - stepZ * j * point_buffer_resolution->x];
				}
			}
			height_value /= LOD * LOD;
			if (height_value >= current_ray_position.y)
			{
				getColorMapIndex(posX, posZ, result);
				return;
			}

			/*Advance ray through the grid*/
			if(tMaxX < tMaxZ)
			{
				if (tMaxX > 2000) LOD = 10; else if (tMaxX > 1000) LOD = 5; else LOD = 1;
				tMaxX += tDeltaX * LOD;
				posX += stepX * LOD;
			}
			else
			{
				if (tMaxZ > 2000) LOD = 10; else if (tMaxZ > 1000) LOD = 5; else LOD = 1;
				tMaxZ += tDeltaZ * LOD;
				posZ += stepZ * LOD;
			}

			/*Check if ray is outside of the grid and going up after max_height is reached*/
			if (posX >= point_buffer_resolution->x || posX < 0 || posZ >= point_buffer_resolution->y || posZ < 0 ||	current_ray_position.y < 0 || current_ray_position.y > max_height && ray_direction.y >= 0)
			{
				return;
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
		int pixel_x, pixel_y, threadId;
		
		pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
		pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
		threadId = pixel_x + pixel_y * texture_resolution->x;
		
		glm::uvec3 color_index(200, 200, 200);
		glm::vec3 ray_direction, ray_position;
		glm::ivec2 pixel_position;

		/*Get the pixel position of this thread*/
		pixel_position = glm::ivec2(pixel_x, pixel_y);

		/*Calculate ray direction and cast ray*/
		ray_direction = *pixel_to_grid_matrix * viewToGridSpace(pixel_position);

		ray_position = ray_direction + *grid_camera_position;
		ray_direction = normalize(ray_direction);
		castRay(ray_position, ray_direction, color_index);
		
		//GL_RGB
		color_buffer[threadId * 3] = color_index.r;
		color_buffer[threadId * 3 + 1] = color_index.g;
		color_buffer[threadId * 3 + 2] = color_index.b;
	}

	/*
	 * Fill empty cells in the GPU grid
	 */
	__global__ void cuda_fillVoid(glm::ivec2 grid_resolution, float* grid, float* temp_grid)
	{
		int threadId, count, posX, posY, tempX, tempY;
		float value;

		posX = blockIdx.x * gridDim.x + threadIdx.x;
		posY = blockIdx.y * gridDim.y + threadIdx.y;
		threadId = posX + posY * point_buffer_resolution->x;
		if (threadId > point_buffer_resolution->x * point_buffer_resolution->y || abs(temp_grid[threadId]) < 0.01f)
			return;

		count = 0;
		value = 0;

		for (int i = 0; i < 7; i++)
		{
			for (int j = 0; j < 7; j++)
			{
				if (i == 3 && j == 3)
					continue;

				tempX = posX - 3 + i;
				tempY = posY - 3 + j;
				if (tempX >= 0 && tempX < grid_resolution.x && tempY >= 0 && tempY < grid_resolution.y)
				{
					count++;
					value += grid[tempX + tempY * grid_resolution.x];
				}
			}
		}
		temp_grid[threadId] = value / count;
	}

	/*
	 * Calculate LOD and save it in the affected cells
	 */
	__global__ void cuda_calculateLOD(glm::ivec2 grid_resolution, float* grid, float* temp_grid, int cells)
	{
		int threadId, count, posX, posY, tempX, tempY;
		float value;

		posX = blockIdx.x * gridDim.x + threadIdx.x;
		posY = blockIdx.y * gridDim.y + threadIdx.y;
		threadId =  posX + posY * point_buffer_resolution->x;
		if (threadId > point_buffer_resolution->x * point_buffer_resolution->y)
			return;
		
		count = 0;
		value = 0;

		for (int i = 0; i < cells; i++)
		{
			for (int j = 0; j < cells; j++)
			{
				tempX = posX + i;
				tempY = posY + j;
				if (tempX < grid_resolution.x && tempY < grid_resolution.y)
				{
					count++;
					value += grid[tempX + tempY * grid_resolution.x];
				}
			}
		}
		temp_grid[threadId] = value/count;
	}

	/*
	* Set device parameters
	*/
	__global__ void cuda_setParameters(glm::vec3 frame_dim, glm::vec3 camera_for, glm::vec3 grid_camera_pos, float max_h)
	{
		*frame_dimension = frame_dim;
		grid_camera_position->y = grid_camera_pos.y;
		max_height = max_h;

		/*Basis change matrix from view to grid space*/
		glm::vec3 u, v, w;
		w = -camera_for;
		u = glm::normalize(glm::cross(glm::vec3(0, 100, 0), w));
		v = glm::cross(w, u);
		*pixel_to_grid_matrix = glm::mat3x3(u,v,w);
	}

	/* 
	 * Initialize device 
	 */
	__global__ void cuda_initializeDeviceVariables(glm::ivec2 point_buffer_res, glm::ivec2 texture_res, float* d_gpu_pointBuffer, unsigned char *d_color_map, glm::ivec2 color_map_res, glm::vec2 cell_siz)
	{
		frame_dimension = new glm::vec3();
		texture_resolution = new glm::ivec2();
		color_map_resolution = new glm::ivec2();
		point_buffer_resolution = new glm::ivec2();
		pixel_to_grid_matrix = new glm::mat3x3();
		cell_size = new glm::vec2();
		grid_camera_position = new glm::vec3(point_buffer_res.x / 2.f * cell_siz.x, 0, point_buffer_res.y / 2.f * cell_siz.y);

		*point_buffer_resolution = point_buffer_res;
		*texture_resolution = texture_res;	
		point_buffer = d_gpu_pointBuffer;
		color_map = d_color_map;
		*color_map_resolution = color_map_res;
		*cell_size = cell_siz;
	}

	/*
	* Free device's variables
	*/
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
	 * Set grid and block dimensions, create LOD, pass parameters to device and call kernels
	 */
	__host__ void rayTrace(glm::ivec2& texture_resolution, glm::vec3& frame_dimensions, glm::vec3& camera_forward, glm::vec3& grid_camera_pos, unsigned char* color_buffer, float max_height)
	{
		/*
		 * Things to consider:
		 *  Branch divergence inside a warp
		 *  Maximum number threads and blocks inside a SM
		 *  Maximum number of threads per block
		 */
		dim3 gridSize, blockSize;

		
		cuda_setParameters << <1, 1 >> > (frame_dimensions, camera_forward, grid_camera_pos, max_height);
		checkCudaErrors(cudaDeviceSynchronize());
		
		blockSize = dim3(1, 480);
		gridSize = dim3(texture_resolution.x / blockSize.x, texture_resolution.y / blockSize.y);
		cuda_rayTrace << <gridSize, blockSize >> > (color_buffer);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	/*
	* Fill cells that do not contain values in grid
	*/
	__host__ void fillVoid(glm::ivec2 & point_buffer_resolution)
	{
		dim3 blockSize, gridSize;
		//TODO: implement
	}

	/*
	* Calculate the LOD in the grid
	*/
	__host__ void calculateLOD(glm::ivec2 & d_grid_resolution, float *d_grid)
	{
		dim3 blockSize, gridSize;
		float* d_temp_grid;

		checkCudaErrors(cudaMalloc(&d_temp_grid, sizeof(float) * d_grid_resolution.x * d_grid_resolution.y));		
		blockSize = dim3(64, 32);
		gridSize = dim3(d_grid_resolution.x / blockSize.x, d_grid_resolution.y / blockSize.y);
		cuda_calculateLOD << <gridSize, blockSize >> > (d_grid_resolution, d_grid, d_temp_grid, 5);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	/*
	 * Initialize variables in the device
	 */
	__host__ void initializeDeviceVariables(glm::ivec2& point_buffer_res, glm::ivec2& texture_res, float* d_gpu_pointBuffer, unsigned char* d_color_map, glm::ivec2& color_map_res, glm::vec2& cell_size)
	{
		cuda_initializeDeviceVariables << <1, 1 >> > (point_buffer_res, texture_res, d_gpu_pointBuffer, d_color_map, color_map_res, cell_size);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	/*
	 * Free memory addresses in device
	 */
	__host__ void freeDeviceVariables()
	{
		cuda_freeDeviceVariables << <1, 1 >> > ();
		checkCudaErrors(cudaDeviceSynchronize());
	}
}
