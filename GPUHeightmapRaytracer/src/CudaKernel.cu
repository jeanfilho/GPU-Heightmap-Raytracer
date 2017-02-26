#include "CudaKernel.cuh"
#include <helper_cuda.h>
#include <iostream>
#include <math_functions.hpp>

namespace CudaSpace
{
	__device__ bool use_color_map = false;
	__device__ float max_height = 0;
	__device__ float *point_buffer;
	__device__ unsigned char *color_map;
	__device__ int LOD_levels, stride_x = 1;
	__device__ glm::vec2 *cell_size;
	__device__ glm::vec3 *frame_dimension;
	__device__ glm::vec3 *camera_forward;
	__device__ glm::vec3 *grid_camera_position;
	__device__ glm::ivec2 *texture_resolution;
	__device__ glm::ivec2 *point_buffer_resolution;
	__device__ glm::ivec2 *color_map_resolution;
	__device__ glm::mat3x3 *pixel_to_grid_matrix;

	/*
	* Get a colormap value from a height map index
	*/
	__device__ void getColorMapValue(int posX, int posZ, glm::uvec3& result)
	{
		int colorX, colorZ;
		colorX = float(posX) / float(point_buffer_resolution->x) * color_map_resolution->x;
		colorZ = float(posZ) / float(point_buffer_resolution->y) * color_map_resolution->y;
		int index = (colorX + colorZ * color_map_resolution->x) * 3;
		result = glm::uvec3(color_map[index], color_map[index + 1], color_map[index + 2]);
	}

	/*
	* Get a value based on max height
	*/
	__device__ void getHeightColorValue(float height, glm::uvec3& result)
	{
		unsigned char r, g, b;
		height = height * 2 / max_height ;
		if(height > 1)
		{
			height -= 1;
			r = 255;
			g = 255 - height * 255;
			b = 0;
		}
		else
		{
			r = 255 * height;
			g = r;
			b = 255 - height * 255;
		}
		result = glm::uvec3(r, g, b);
	}

	/*
	 * GLM::SIGN produces compile error in CUDA
	 */
	__device__ char sign(float input)
	{
		return input >= 0 ? 1 : -1;
	}

	/*
	 * Set up parameters for tracing a single ray at the beginning
	 */
	__device__ void setUpStartingParameters(float &tMax, float &tDelta, char &step, int &pos, float ray_origin, float ray_direction, float cell_size)
	{
		/*Set starting voxel*/
		pos = static_cast<int>(floor(ray_origin / cell_size));

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
			tMax = ((floor(ray_origin / cell_size) + (ray_direction < 0 ? 0 : 1)) * cell_size - ray_origin) / ray_direction;
			tDelta = (ray_direction < 0 ? -1 : 1) * cell_size / ray_direction;
		}
	}

	/*
	 * Update raytracing parameters accourdingly to the current LOD
	 */
	__device__ void updateParameters(float &tMax, int &pos, float ray_position, float ray_direction, float cell_size)
	{
		pos *= 2;
		tMax = ((floor(ray_position / cell_size) + (ray_direction < 0 ? 0 : 1)) * cell_size - ray_position) / ray_direction;
	}

	/*
	 * Retrieve the height value from point buffer based on LOD and position
	 */
	__device__ float getPointBufferValue(int posX, int posZ, int LOD)
	{
		int index;

		index = posX + point_buffer_resolution->x * stride_x * posZ;

		return point_buffer[index];
	}

	/*
	 *	Cast a ray through the grid {Amanatides, 1987 #22}
	 *	ray_direction MUST be normalized
	 *	
	 *	tMax: distance advanced in X and Z axis - at start, this represents the distance until the first intersection
	 *	tDelta: distance advanced when moving one cell in the grid
	 */
	__device__ void castRay(glm::vec3& ray_origin, glm::vec3& ray_direction, glm::uvec3& result)
	{
		float tMaxX, tMaxZ, tDeltaX, tDeltaZ, height_value;
		int posX, posZ, LOD;
		char stepX, stepZ;
		glm::vec3 ray_position_entry, ray_position_exit;

		LOD = 0;
		setUpStartingParameters(tMaxX, tDeltaX, stepX, posX, ray_origin.x, ray_direction.x, cell_size->x * pow(2.0f, LOD));
		setUpStartingParameters(tMaxZ, tDeltaZ, stepZ, posZ, ray_origin.z, ray_direction.z, cell_size->y * pow(2.0f, LOD));

		/*Loop the DDA algorithm*/
		ray_position_entry = ray_origin;
		ray_position_exit = ray_origin + ray_direction * (tMaxX < tMaxZ ? tMaxX : tMaxZ);
		while(true)
		{
			/*Check if ray intersects with back faces - Texel intersection from Dick, C., et al. (2009). GPU ray-casting for scalable terrain rendering. Proceedings of EUROGRAPHICS, Citeseer. Page */
			height_value = getPointBufferValue(posX, posZ, LOD);
			while ((ray_direction.y >= 0 ? ray_position_entry.y : ray_position_exit.y) <= height_value)
			{
				/*Step one LOD down*/
				if (LOD > 1)
				{
					ray_position_entry += glm::max(0.0f, (height_value - ray_position_entry.y) / ray_direction.y) * ray_direction;

					LOD--;
					tDeltaX /= 2;
					tDeltaZ /= 2;
					updateParameters(tMaxX, posX, ray_position_entry.x, ray_direction.x, cell_size->x * pow(2.0f, LOD));
					updateParameters(tMaxZ, posZ, ray_position_entry.z, ray_direction.z, cell_size->y * pow(2.0f, LOD));

					ray_position_exit = ray_origin + ray_direction * (tMaxX < tMaxZ ? tMaxX : tMaxZ);
					height_value = getPointBufferValue(posX, posZ, LOD);
				}
				else
				{
					if (use_color_map)
						getColorMapValue(posX, posZ, result);
					else
						getHeightColorValue(height_value, result);
					return;
				}
			}

			/*Advance ray through the grid*/
			ray_position_entry = ray_position_exit;
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
			ray_position_exit = ray_origin + ray_direction * (tMaxX < tMaxZ ? tMaxX : tMaxZ); //Check where next step will be
			

			/*Check if ray is outside of the grid/quadtree cell or going up after max_height is reached*/
			if(LOD < LOD_levels - 2)
			{
				/*Step one LOD up*/
				if(true)
				{
					LOD++;
					updateParameters(tMaxX, posX, ray_position_exit.x, ray_direction.x, cell_size->x * pow(2.0f, LOD));
					updateParameters(tMaxZ, posZ, ray_position_exit.z, ray_direction.z, cell_size->y * pow(2.0f, LOD));
				}
			}
			else if (posX >= point_buffer_resolution->x || posX < 0 || posZ >= point_buffer_resolution->y || posZ < 0 ||	ray_position_entry.y < 0 || ray_position_entry.y > max_height && ray_direction.y >= 0)
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
		
		glm::uvec3 color_value(200, 200, 200);
		glm::vec3 ray_direction, ray_position;
		glm::ivec2 pixel_position;

		/*Get the pixel position of this thread*/
		pixel_position = glm::ivec2(pixel_x, pixel_y);

		/*Calculate ray direction and cast ray*/
		ray_direction = *pixel_to_grid_matrix * viewToGridSpace(pixel_position);

		ray_position = ray_direction + *grid_camera_position;
		ray_direction = normalize(ray_direction);
		castRay(ray_position, ray_direction, color_value);
		
		//GL_RGB
		color_buffer[threadId * 3] = color_value.r;
		color_buffer[threadId * 3 + 1] = color_value.g;
		color_buffer[threadId * 3 + 2] = color_value.b;
	}

	/*
	* Set device parameters
	*/
	__global__ void cuda_setParameters(glm::vec3 frame_dim, glm::vec3 camera_for, glm::vec3 grid_camera_pos, bool use_color)
	{
		*frame_dimension = frame_dim;
		grid_camera_position->y = grid_camera_pos.y;
		use_color_map = use_color;

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
	__global__ void cuda_initializeDeviceVariables(glm::ivec2 point_buffer_resolution, glm::ivec2 texture_resolution, float* point_buffer, unsigned char *color_map, glm::ivec2 color_map_resolution, glm::vec2 cell_size, int LOD_levels, int stride_x, float max_height)
	{
		CudaSpace::texture_resolution = new glm::ivec2();
		CudaSpace::color_map_resolution = new glm::ivec2();
		CudaSpace::point_buffer_resolution = new glm::ivec2();
		CudaSpace::cell_size = new glm::vec2();

		frame_dimension = new glm::vec3();
		pixel_to_grid_matrix = new glm::mat3x3();
		grid_camera_position = new glm::vec3(point_buffer_resolution.x / 2.f * cell_size.x, 0, point_buffer_resolution.y / 2.f * cell_size.y);

		*CudaSpace::point_buffer_resolution = point_buffer_resolution;
		*CudaSpace::texture_resolution = texture_resolution;
		CudaSpace::point_buffer = point_buffer;
		CudaSpace::color_map = color_map;
		*CudaSpace::color_map_resolution = color_map_resolution;
		*CudaSpace::cell_size = cell_size;
		CudaSpace::LOD_levels = LOD_levels;
		CudaSpace::stride_x = stride_x;
		CudaSpace::max_height = max_height;
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
	__host__ void rayTrace(glm::ivec2& texture_resolution, glm::vec3& frame_dimensions, glm::vec3& camera_forward, glm::vec3& grid_camera_pos, unsigned char* color_buffer, bool use_color_map)
	{
		/*
		 *  Things to consider:
		 *  Branch divergence inside a warp
		 *  Maximum number threads and blocks inside a SM
		 *  Maximum number of threads per block
		 */
		dim3 gridSize, blockSize;

		
		cuda_setParameters << <1, 1 >> > (frame_dimensions, camera_forward, grid_camera_pos, use_color_map);
		checkCudaErrors(cudaDeviceSynchronize());
		
		blockSize = dim3(1, 1080/2);
		// ReSharper disable CppAssignedValueIsNeverUsed
		gridSize = dim3(texture_resolution.x / blockSize.x, texture_resolution.y / blockSize.y);
		cuda_rayTrace << <gridSize, blockSize >> > (color_buffer);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	/*
	 * Initialize variables in the device
	 */
	__host__ void initializeDeviceVariables(glm::ivec2& point_buffer_res, glm::ivec2& texture_res, float* d_gpu_pointBuffer, unsigned char* d_color_map, glm::ivec2& color_map_res, glm::vec2& cell_size, int LOD_levels, int stride_x, float max_height)
	{
		cuda_initializeDeviceVariables << <1, 1 >> > (point_buffer_res, texture_res, d_gpu_pointBuffer, d_color_map, color_map_res, cell_size, LOD_levels, stride_x, max_height);
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
