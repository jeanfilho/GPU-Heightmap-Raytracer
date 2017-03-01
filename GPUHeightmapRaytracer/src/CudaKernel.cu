#include "CudaKernel.cuh"
#include <helper_cuda.h>
#include <iostream>
#include <math_functions.hpp>

namespace CudaSpace
{
	__device__ bool use_color_map = false;
	__device__ float max_height = 0;
	__device__ float *point_buffer;
	__device__ int *LOD_indexes, *LOD_resolutions;
	__device__ CudaSpace::Color *color_map;
	__device__ int LOD_levels, stride_x = 1;
	__device__ glm::vec3 *frame_dimension;
	__device__ glm::vec3 *camera_forward;
	__device__ glm::vec3 *grid_camera_position;
	__device__ glm::ivec2 *texture_resolution;
	__device__ glm::ivec2 *point_buffer_resolution;
	__device__ glm::ivec2 *boundary;
	__device__ glm::mat3x3 *pixel_to_grid_matrix;

	/*
	* Get a colormap value from a height map index
	*/
	__device__ void getColorMapValue(int posX, int posZ, bool mirrorX, bool mirrorZ, Color& result)
	{
		if (mirrorX)
			posX = LOD_resolutions[0] - 1 - posX;
		if (mirrorZ)
			posZ = LOD_resolutions[0] - 1 - posZ;

		result = color_map[posX + posZ * LOD_resolutions[0]];
	}

	/*
	* Get a value based on max height
	*/
	__device__ void getHeightColorValue(float height, Color& result)
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
		result = Color(r, g, b);
	}

	/*
	 * Retrieve the height value from point buffer based on LOD and position
	 */
	__device__ float getPointBufferValue(int posX, int posZ, bool mirrorX, bool mirrorZ, int LOD)
	{
		if (mirrorX)
			posX = LOD_resolutions[LOD] - 1 - posX;
		if (mirrorZ)
			posZ = LOD_resolutions[LOD] - 1 - posZ;

		return point_buffer[LOD_indexes[LOD] + posX + posZ * LOD_resolutions[LOD]];
	}

	/*
	 *Calculate exit point based on current ray position
	 */
	__device__ void calculateExitPointAndEdge(glm::vec3& entry, glm::vec3& direction, glm::vec3& exit, int &edge, int LOD)
	{
		float tX, tZ;
		tX = ((floor(entry.x / pow(2.f, LOD)) + 1) * pow(2.f, LOD) - entry.x) / direction.x;
		tZ = ((floor(entry.z / pow(2.f, LOD)) + 1) * pow(2.f, LOD) - entry.z) / direction.z;
		if(tX <= tZ)
		{
			exit = entry + tX * direction;
			exit.x = (floor(entry.x / pow(2.f, LOD)) + 1) * pow(2.f, LOD);
			edge = floor(exit.x / pow(2.f, LOD));
		}
		else
		{
			exit = entry + tZ * direction;
			exit.z = (floor(entry.z / pow(2.f, LOD)) + 1) * pow(2.f, LOD);
			edge = floor(exit.z / pow(2.f, LOD));
		}
	}

	/*
	 * Test if the ray intersects with the height field
	 */
	__device__ bool testIntersection(glm::vec3 &entry, glm::vec3 &exit, glm::vec3 &direction, bool mirrorX, bool mirrorZ, int &LOD)
	{
		bool result;
		float height;

		height = getPointBufferValue(floor(entry.x / pow(2.f, LOD)), floor(entry.z / pow(2.f, LOD)), mirrorX, mirrorZ, LOD);
		if(direction.y >= 0)
		{
			result = entry.y <= height;
		}
		else
		{
			result = exit.y <= height;
			if (result)
				entry += glm::max(0.f, (height - entry.y) / direction.y) * direction;
		}

		return result;		
	}


	/*
	 *	Dick, C., et al. (2009). GPU ray-casting for scalable terrain rendering. Proceedings of EUROGRAPHICS, Citeseer.
	 *	ray_direction MUST be normalized
	 */
	__device__ void castRay(glm::vec3& ray_position, glm::vec3& ray_direction, Color& result)
	{
		bool mirrorX, mirrorZ;
		glm::vec3 ray_exit;
		int edge;
		int LOD = LOD_levels - 1;
		bool intersection;

		/*Mirror direction to simplify algorithm*/
		if(ray_direction.x < 0)
		{
			mirrorX = true;
			ray_direction.x = -ray_direction.x;
			ray_position.x = point_buffer_resolution->x * pow(2.f, LOD) - ray_position.x;
		}
		else
		{
			mirrorX = false;
		}

		if(ray_direction.z < 0)
		{
			mirrorZ = true;
			ray_direction.z = -ray_direction.z;
			ray_position.z = point_buffer_resolution->y * pow(2.f, LOD) - ray_position.z;
		}
		else
		{
			mirrorZ = false;
		}

		/*Advance ray until it is outside of the buffer*/
		while(ray_position.x < boundary->x && ray_position.z < boundary->y && !(ray_direction.y > 0 && ray_position.y > max_height))
		{
			calculateExitPointAndEdge(ray_position, ray_direction, ray_exit, edge, LOD);
			intersection = testIntersection(ray_position, ray_exit, ray_direction, mirrorX, mirrorZ, LOD);
			if(intersection)
			{
				if (LOD > 0)
					LOD--;
				else
				{
					if (use_color_map)
						getColorMapValue(floor(ray_position.x), floor(ray_position.z), mirrorX, mirrorZ, result);
					else
						getHeightColorValue(ray_position.y, result);
					return;
				}
				
			}
			else
			{
				LOD = glm::min(LOD + 1 - (edge % 2), LOD_levels - 1);
				ray_position = ray_exit;			
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
		
		Color color_value(static_cast<unsigned char>(200), static_cast<unsigned char>(200), static_cast<unsigned char>(200));
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
	__global__ void cuda_setParameters(glm::vec3 frame_dim, glm::vec3 camera_for, glm::vec3 grid_camera_pos, bool use_color, float max_height)
	{
		*frame_dimension = frame_dim;
		*grid_camera_position = grid_camera_pos;
		use_color_map = use_color;
		CudaSpace::max_height = max_height;

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
	__global__ void cuda_initializeDeviceVariables(glm::ivec2 point_buffer_resolution, glm::ivec2 texture_resolution, float* point_buffer, CudaSpace::Color *color_map, int LOD_levels, int stride_x, float max_height)
	{
		CudaSpace::texture_resolution = new glm::ivec2();
		CudaSpace::point_buffer_resolution = new glm::ivec2();

		LOD_indexes = new int[LOD_levels]();
		LOD_resolutions = new int[LOD_levels]();
		LOD_resolutions[LOD_levels - 1] = point_buffer_resolution.x;
		LOD_indexes[LOD_levels - 1] = 0;
		for(auto i = LOD_levels - 2; i >= 0; i--)
		{
			LOD_indexes[i] = LOD_indexes[i + 1] + LOD_resolutions[i + 1] * LOD_resolutions[i + 1];
			LOD_resolutions[i] = LOD_resolutions[i + 1] * 2;
		}

		boundary = new glm::ivec2(LOD_resolutions[0], LOD_resolutions[0]);
		frame_dimension = new glm::vec3();
		pixel_to_grid_matrix = new glm::mat3x3();
		grid_camera_position = new glm::vec3();

		*CudaSpace::point_buffer_resolution = point_buffer_resolution;
		*CudaSpace::texture_resolution = texture_resolution;
		CudaSpace::point_buffer = point_buffer;
		CudaSpace::color_map = color_map;
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
		delete(point_buffer_resolution);
		delete[](LOD_indexes);
		delete[](LOD_resolutions);
	}

	/*
	 * Set grid and block dimensions, create LOD, pass parameters to device and call kernels
	 */
	__host__ void rayTrace(glm::ivec2& texture_resolution, glm::vec3& frame_dimensions, glm::vec3& camera_forward, glm::vec3& grid_camera_pos, unsigned char* color_buffer, bool use_color_map, float max_height)
	{
		/*
		 *  Things to consider:
		 *  Branch divergence inside a warp
		 *  Maximum number threads and blocks inside a SM
		 *  Maximum number of threads per block
		 */
		dim3 gridSize, blockSize;
		cuda_setParameters << <1, 1 >> > (frame_dimensions, camera_forward, grid_camera_pos, use_color_map, max_height);
		checkCudaErrors(cudaDeviceSynchronize());
		
		blockSize = dim3(1, texture_resolution.y/2);
		// ReSharper disable CppAssignedValueIsNeverUsed
		gridSize = dim3(texture_resolution.x / blockSize.x, texture_resolution.y / blockSize.y);
		cuda_rayTrace << <gridSize, blockSize >> > (color_buffer);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	/*
	 * Initialize variables in the device
	 */
	__host__ void initializeDeviceVariables(glm::ivec2& point_buffer_res, glm::ivec2& texture_res, float* d_gpu_pointBuffer, CudaSpace::Color* d_color_map, int LOD_levels, int stride_x, float max_height)
	{
		cuda_initializeDeviceVariables << <1, 1 >> > (point_buffer_res, texture_res, d_gpu_pointBuffer, d_color_map, LOD_levels, stride_x, max_height);
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
