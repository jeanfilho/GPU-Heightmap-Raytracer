/***********************************************************
* A Template for building OpenGL applications using GLUT
*
* Author: Perspective @ cprogramming.com
* Date : Jan, 2005
*
* Description:
* This code initializes an OpenGL ready window
* using GLUT. Some of the most common callbacks
* are registered to empty or minimal functions.
*
* This code is intended to be a quick starting point
* when building GLUT applications.
*
* Source: http://www.cprogramming.com/snippets/source-code/a-code-template-for-opengl-divide-glut
*
***********************************************************/

#define GLM_FORCE_CUDA

#include <iostream>
#include <fstream>
#include <thread>
#include <string>
#include <chrono>
#include <thread>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtx/rotate_vector.hpp>

#include "jpeglib.h"

#include <liblas/liblas.hpp>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "CudaKernel.cuh"
#include "helper_cuda.h"


//============================
//		GLOBAL VARIABLES
//============================

// Filenames
std::string point_cloud_file = "points.las";
std::string color_map_file = "autzen.jpg";

// Camera related
glm::ivec2 texture_resolution(1920, 1080);
glm::vec3
	camera_position(0, 0, 0),
	camera_forward(glm::normalize(glm::vec3(5, -.9, 0))),
	frame_dimension(40, 30, 30); //width, height, distance from camera
glm::vec2 boundaries(0, 0);

GLuint textureID;
GLuint bufferID;
bool use_LOD = false;
bool use_color_map = false;

// JPEG image
unsigned char *h_color_map = NULL;
glm::ivec2 color_map_resolution = glm::zero<glm::ivec2>();

// Point buffer to be copied to GPU
float* h_point_buffer;
glm::ivec2 point_buffer_resolution(512, 512);

// CPU-Side point sections
const int point_sections_size = 4;
std::thread* thread_pool[point_sections_size][point_sections_size];
bool thread_exit[point_sections_size][point_sections_size];
float* point_sections[point_sections_size][point_sections_size];
glm::vec2 point_sections_origins[point_sections_size][point_sections_size];
glm::vec2 cell_size; //Cell size at the finest LOD level
float max_height = 0;
float height_tolerance = 10;
const int LOD_levels = 4;
const int stride_x = static_cast<int>(glm::pow(4, LOD_levels - 1)) + 1; // Number of elements per Quad-tree root

// clock
std::chrono::system_clock sys_clock;
std::chrono::time_point<std::chrono::system_clock> last_frame, current_frame;
std::chrono::duration<float> delta_time;

// movement
float movement_rht = 0;
float movement_fwd = 0;
float movement_up = 0;
float right_movement = 0;
float wasd_movement_distance = 250;
float qe_movement_distance = 500;

// rotation
float rotation_up = 0;
float rotation_right = 0;
float ik_rotation_angle = 1.f;
float jl_rotation_angle = 1.f;

//============================
//		CUDA VARIABLES
//============================

float* d_point_buffer;
unsigned char *d_color_map = NULL;
struct cudaGraphicsResource* cuda_pbo_resource;

//============================
//		HELPER FUNCTIONS
//============================

/*Loads JPEG image from disk*/
bool loadJPEG(std::string filename)
{
	unsigned char *pos;
	unsigned char r, g, b;
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	int width, height;
	JSAMPARRAY pJpegBuffer;

	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);

	/*Open JPEG and start decompression*/
	int row;
	FILE * infile;
	if (fopen_s(&infile, ("../Data/" + filename).c_str(), "rb") != NULL) 
	{
			fprintf(stderr, "can't open %s\n", filename.c_str());
			return false;
	}

	jpeg_stdio_src(&cinfo, infile);
	jpeg_read_header(&cinfo, TRUE);
	jpeg_start_decompress(&cinfo);
	
	width = cinfo.output_width;
	height = cinfo.output_height;
	h_color_map = new unsigned char[width*height * 3];
	pos = h_color_map + width * height * 3;

	row = width * cinfo.output_components;
	pJpegBuffer = (*cinfo.mem->alloc_sarray)
		(reinterpret_cast<j_common_ptr>(&cinfo), JPOOL_IMAGE, row, 1);

	/*Read data in JPEG - scan happens from top to bottom*/
	while (cinfo.output_scanline < cinfo.output_height)
	{
		jpeg_read_scanlines(&cinfo, pJpegBuffer, 1);
		pos -= row;
		for (int x = 0; x < width; x++)
		{
			r = pJpegBuffer[0][cinfo.output_components * x];
			g = pJpegBuffer[0][cinfo.output_components * x + 1];
			b = pJpegBuffer[0][cinfo.output_components * x + 2];

			*(pos++) = r;
			*(pos++) = g;
			*(pos++) = b;
		}
		pos -= row;
	}

	/*Finish decompression and release object*/
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(infile);

	color_map_resolution = glm::ivec2(width, height);

	/*Pass values to GPU*/
	size_t count = sizeof(unsigned char) * 3 * width * height;
	checkCudaErrors(cudaMalloc(&d_color_map, count));
	checkCudaErrors(cudaMemcpy(d_color_map, h_color_map, count, cudaMemcpyHostToDevice));
	
	return true;
}


//============================
//		LAS FUNCTIONS
//============================

/*
 * Read LAS header before starting the ray tracing and collect necessary information
 * Set the camera position to the center of the point cloud
 * Source: http://www.liblas.org/tutorial/cpp.html
 */
void readLASHeader(std::string filename)
{
	/*Create input stream and associate it with .las file opened to read in binary mode*/
	std::ifstream ifs;
	ifs.open("../Data/" + filename, std::ios::in | std::ios::binary);
	if (!ifs.is_open())
	{
		std::cout << "Error opening " + filename << std::endl;
		exit(1);
	}

	/*Create a ReaderFactory and instantiate a new liblas::Reader using the stream.*/
	liblas::ReaderFactory f;
	liblas::Reader reader = f.CreateWithStream(ifs);

	/*After the reader has been created, you can access members of the Public Header Block*/
	liblas::Header const& header = reader.GetHeader();
	std::cout << "LAS File Loaded." << std::endl;
	std::cout << "Compressed: " << (header.Compressed() == true) << std::endl;
	std::cout << "Points count: " << header.GetPointRecordsCount() << std::endl;
	std::cout << "MinX: " << header.GetMinX() << " MinY: " << header.GetMinY() << " MinZ: " << header.GetMinZ() << std::endl;
	std::cout << "MaxX: " << header.GetMaxX() << " MaxY: " << header.GetMaxY() << " MaxZ: " << header.GetMaxZ() << std::endl;
	std::cout << "ScaleX: " << header.GetScaleX() << " ScaleY: " << header.GetScaleY() << " ScaleZ: " << header.GetScaleZ() << std::endl;
	std::cout << "OffsetX: " << header.GetOffsetX() << " OffsetY: " << header.GetOffsetY() << " OffsetZ: " << header.GetOffsetZ() << std::endl;
	double deltaX, deltaY;
	deltaX = header.GetMaxX() - header.GetMinX();
	deltaY = header.GetMaxY() - header.GetMinY();
	std::cout << "DiffX: " << deltaX << " DiffY: " << deltaY << std::endl;

	/*Calculate area per point to set cell dimension*/
	float point_area = float(deltaX * deltaY / header.GetPointRecordsCount());
	cell_size.x = point_area;
	cell_size.y = point_area;

	/*Place the camera on the center of the point cloud*/
	camera_position = glm::vec3(deltaX / 2, header.GetMaxZ() - header.GetMinZ(), deltaY/2);

	/*Close the file stream*/
	ifs.close();
}

/*
* Load points using libLas Library in LAS format
* Source: http://www.liblas.org/tutorial/cpp.html
*/
void loadPointDataLASToGrid(std::string filename, glm::ivec2 section_position)
{
	/*Create input stream and associate it with .las file opened to read in binary mode*/
	std::ifstream ifs;
	ifs.open("../Data/" + filename, std::ios::in | std::ios::binary);
	if (!ifs.is_open())
	{
		std::cout << "Error opening " + filename << std::endl;
		exit(1);
	}

	/*Create a ReaderFactory and instantiate a new liblas::Reader using the stream.*/
	liblas::ReaderFactory f;
	liblas::Reader reader = f.CreateWithStream(ifs);

	/*After the reader has been created, you can access members of the Public Header Block*/
	liblas::Header const& header = reader.GetHeader();

	/*Define grid boundaries*/
	glm::vec2 lower_boundary, higher_boundary;
	lower_boundary = point_sections_origins[section_position.x][section_position.y];
	higher_boundary = point_sections_origins[section_position.x][section_position.y] + static_cast<float>(point_buffer_resolution.x) * cell_size;

	/*Iterate through point records and calculate the height contribution to each neighboring grid cell*/
	while (reader.ReadNextPoint())
	{
		liblas::Point const& p = reader.GetPoint();
		int x, y, // X and Y coordinates in the finest grid
			temp_x, temp_y;
		unsigned int index[LOD_levels];
		float fX, fY, fZ;

		fX = float(p.GetX() - header.GetMinX());
		fY = float(p.GetY() - header.GetMinY());
		fZ = float(p.GetZ() - header.GetMinZ());

		/* Skip if the point is outside of the grid */
		if (fX < lower_boundary.x || fX >= higher_boundary.x || fY < lower_boundary.y || fY >= higher_boundary.y)
			continue;

		/* Calculate point position in grid */
		x = static_cast<int>(glm::floor((fX - point_sections_origins[section_position.x][section_position.y].x) / cell_size.x));
		y = static_cast<int>(glm::floor((fY - point_sections_origins[section_position.x][section_position.y].y) / cell_size.y));

		/* Calculate offsets in grid */
		temp_x = x / static_cast<int>(glm::pow(2, LOD_levels - 1));
		temp_y = y / static_cast<int>(glm::pow(2, LOD_levels - 1));
		index[LOD_levels - 1] = temp_x * stride_x + temp_y * point_buffer_resolution.x * stride_x;
		for (int i = LOD_levels - 2; i >= 0; i--)
		{
			temp_x = x % static_cast<int>(glm::pow(2, i + 1)) / static_cast<int>(glm::pow(2, i)); // results in either 0 or 1
			temp_y = x % static_cast<int>(glm::pow(2, i + 1)) / static_cast<int>(glm::pow(2, i));
			index[i] = index[i + 1] + 1 + (temp_x * (static_cast<int>(glm::pow(4, i - 1)) + 1) + temp_y * 2 * static_cast<int>(glm::pow(4, i - 1)) + 1);
		}

		/* Break the loop if the thread must be terminated */
		if (thread_exit[section_position.x][section_position.y])
			break;

		/*Insert the highest values from finest to coarsest level of the Quad-tree*/
		for (int i = 0; i < LOD_levels; i++)
		{
			if (*(point_sections[section_position.x][section_position.y] + index[i]) <= fZ)
				*(point_sections[section_position.x][section_position.y] + index[i]) = fZ;
			else
				break;
		}

		/*Set the highest value to visualize it later*/
		if (max_height < fZ && fZ - max_height < height_tolerance)
		{
			max_height = fZ;
		}
	}

	/*Close the file stream*/
	ifs.close();
}


//============================
//		SECTION AND GRID
//          FUNCTIONS
//============================

/*
 * Allocate a grid section for the out-of-core functionality
 * The quad-tree piramid is allocate contiguously to facilitate the copy of a section
 */
void allocateSection(glm::ivec2 pos, glm::vec2 origin)
{
	point_sections_origins[pos.x][pos.y] = origin;
	point_sections[pos.x][pos.y] = new float[sizeof(float) * point_buffer_resolution.x * point_buffer_resolution.y * (static_cast<int>(glm::pow(4, LOD_levels-1) + 1))]();
	thread_pool[pos.x][pos.y] = new std::thread(loadPointDataLASToGrid, point_cloud_file, pos);;
}

/* 
 * Spawn threads at the initialization phase to start loading points 
 * The camera starts at the center of the sections grid (readLASHeader() must be called before this function)
 */
void initializeSections()
{
	for(int i = 0; i < point_sections_size; i++)
	{
		for (int j = 0; j < point_sections_size; j++)
		{
			allocateSection(glm::ivec2(i, j),
				glm::vec2(camera_position.x, camera_position.y) -
				glm::vec2((i - static_cast<float>(point_sections_size) / 2.0f) * cell_size.x * static_cast<float>(point_buffer_resolution.x),
						  (j - static_cast<float>(point_sections_size) / 2.0f) * cell_size.y * static_cast<float>(point_buffer_resolution.y)));
		}
	}

}

/*
 * Helper function for manageSections
 */
void unloadSectionsColumn(int column)
{
	for(int i = 0; i < point_sections_size; i++)
	{
		if (thread_pool[column][i]->joinable())
		{
			thread_exit[column][i] = true;
			thread_pool[column][i]->join();
			delete thread_pool[column][i];
			thread_exit[column][i] = false;
		}
		delete[]point_sections[column][i];
	}
}


/*
* Helper function for manageSections
*/
void unloadSectionsRow(int row)
{
	for (int i = 0; i < point_sections_size; i++)
	{
		if (thread_pool[i][row]->joinable())
		{
			thread_exit[i][row] = true;
			thread_pool[i][row]->join();
			delete thread_pool[i][row];
			thread_exit[i][row] = false;
		}
		delete[]point_sections[i][row];
	}
}

/*
* Move sections X cells horizontally
* + is RIGHT
*/
void rearrangeSectionsX(int x)
{
	int i, j;
	if (x >= 0)
	{
		for (i = point_sections_size - 1; i >= x; i--)
			for (j = 0; j < point_sections_size; j++)
			{
				point_sections[i][j] = point_sections[i - x][j];
				point_sections_origins[i][j] = point_sections_origins[i - x][j];
				thread_pool[i][j] = thread_pool[i - x][j];
			}
	}
	else
	{
		for (i = 0; i < point_sections_size - x; i++)
			for (j = 0; j < point_sections_size; j++)
			{
				point_sections[i][j] = point_sections[i - x][j];
				point_sections_origins[i][j] = point_sections_origins[i - x][j];
				thread_pool[i][j] = thread_pool[i - x][j];
			}
	}
}

/*
* Move sections Y cells horizontally
* + is DOWN
*/
void rearrangeSectionsY(int y)
{
	int i, j;
	if (y >= 0)
	{
		for (i = 0; i < point_sections_size; i++)
			for (j = point_sections_size - 1; j >= y; j--)
			{
				point_sections[i][j] = point_sections[i][j - y];
				point_sections_origins[i][j] = point_sections_origins[i][j - y];
				thread_pool[i][j] = thread_pool[i][j - y];
			}
	}
	else
	{
		for (i = 0; i < point_sections_size; i++)
			for (j = 0; j < point_sections_size - y; j++)
			{
				point_sections[i][j] = point_sections[i][j - y];
				point_sections_origins[i][j] = point_sections_origins[i][j - y];
				thread_pool[i][j] = thread_pool[i][j - y];
			}
	}
}

/* 
 * Based on camera position, load and unload point sections
 * If the camera's grid is less than the set distance to a border, rearrange the grid
 */
void manageSections()
{
	float distance;
	distance = static_cast<float>(point_buffer_resolution.x) * cell_size.x;

	/*Allocate left - move sections right*/
	if (camera_position.x < point_sections_origins[0][0].x + distance)
	{
		unloadSectionsColumn(point_sections_size - 1);
		rearrangeSectionsX(1);
		for (int i = 0; i < point_sections_size; i++)
			allocateSection(glm::ivec2(0, i),
				point_sections_origins[1][i] - glm::vec2(1, 0) * static_cast<float>(point_buffer_resolution.x) * cell_size.x);
	}

	/*Allocate right - Move sections left*/
	if (camera_position.x >= point_sections_origins[point_sections_size - 1][point_sections_size - 1].x + distance)
	{
		unloadSectionsColumn(0);
		rearrangeSectionsX(-1);
		for (int i = 0; i < point_sections_size; i++)
			allocateSection(glm::ivec2(point_sections_size - 1, i),
				point_sections_origins[point_sections_size - 2][i] + glm::vec2(1, 0) * static_cast<float>(point_buffer_resolution.x) * cell_size.x);
	}

	/*Allocate up - move sections down*/
	if (camera_position.y < point_sections_origins[0][0].y + distance)
	{
		unloadSectionsRow(0);
		rearrangeSectionsY(-1);
		for (int i = 0; i < point_sections_size; i++)
			allocateSection(glm::ivec2(i, point_sections_size - 1),
				point_sections_origins[i][point_sections_size - 2] + glm::vec2(0, 1) * static_cast<float>(point_buffer_resolution.y) * cell_size.y);
	}

	/*Allocate down - move sections up*/
	if (camera_position.y >= point_sections_origins[point_sections_size - 1][point_sections_size - 1].y + distance)
	{
		unloadSectionsRow(point_sections_size - 1);
		rearrangeSectionsY(1);
		for (int i = 0; i < point_sections_size; i++)
			allocateSection(glm::ivec2(i, 0),
				point_sections_origins[i][1] - glm::vec2(0, 1) * static_cast<float>(point_buffer_resolution.y) * cell_size.y);
	}
}


/*
* Set up CPU-Side buffer that is going to be transferred over to the GPU
*
* NOTE: it is more efficient to pre-allocate the point
* buffer in the RAM and then pass it to the GPU
* than passing every line at a time
*
*/
void copyPointBuffer()
{	
	glm::vec3 bottom_left, top_right, offset;
	int minX, maxX, minY, maxY;

	/*Set the corners of the point buffer*/
	offset = glm::vec3(cell_size.x * point_buffer_resolution.x / 2.0f, 0, cell_size.y * point_buffer_resolution.y / 2.0f);
	bottom_left = camera_position - offset;
	top_right = camera_position + offset - glm::vec3(FLT_MIN, 0, FLT_MIN); //subtract an amount in case the camera is at the center of a grid 

	/*Left section index*/
	minX = 0;
	while(bottom_left.x > point_sections_origins[minX][0].x)
	{
		minX++;
	}
	minX--;

	/*Right section index*/
	maxX = 0;
	while(top_right.x > point_sections_origins[maxX][0].x)
	{
		maxX++;
	}
	maxX--;

	/*Top section index*/
	maxY = 0;
	while (top_right.z > point_sections_origins[0][maxY].y)
	{
		maxY++;
	}
	maxY--;

	/*Bottom section index*/
	minY = 0;
	while (top_right.z > point_sections_origins[0][minY].y)
	{
		minY++;
	}
	minY--;


	/*Copy the quad-trees into the point buffer, start with lower left corner and proceed row-wise */
	glm::ivec2 cell_position, section_position;
	int row_index, row_offset;

	/*Cell position at lower left section*/
	section_position = bottom_left - glm::vec3(point_sections_origins[minX][minY].x, 0, point_sections_origins[minX][minY].y);
	cell_position = glm::ivec2(static_cast<int>(glm::floor(section_position.x / cell_size.x)), static_cast<int>(glm::floor(section_position.y / cell_size.y)));

	/*Copy the data from the lower left section*/
	row_offset = 0;
	for (row_index = cell_position.y; row_index < point_buffer_resolution.y; row_index++)
	{
		memcpy(h_point_buffer + row_offset * stride_x * point_buffer_resolution.x,
			point_sections[minX][minY] + cell_position.x * stride_x + row_index * point_buffer_resolution.x * stride_x,
			sizeof(float) * stride_x * (point_buffer_resolution.x - cell_position.x));

		row_offset++;
	}

	/*Copy the data from the bottom right section*/
	row_offset = 0;
	row_index = cell_position.x == 0 ? point_buffer_resolution.y : cell_position.y;
	for (row_index; row_index < point_buffer_resolution.y; row_index++)
	{
		memcpy(h_point_buffer + (point_buffer_resolution.x - cell_position.x) * stride_x + row_offset * stride_x * point_buffer_resolution.x,
			point_sections[maxX][minY] + row_index * point_buffer_resolution.x * stride_x,
			sizeof(float) * stride_x * cell_position.x);

		row_offset++;
	}

	/*Copy the data from top left section */
	row_offset = 0;
	row_index = cell_position.y == 0 ? cell_position.y : 0;
	for (row_index; row_index < cell_position.y; row_index++)
	{
		memcpy(h_point_buffer + row_index * stride_x * point_buffer_resolution.x,
			point_sections[minX][maxY] + cell_position.x * stride_x + row_offset * stride_x * point_buffer_resolution.x,
			sizeof(float) * stride_x * (point_buffer_resolution.x - cell_position.x));

		row_offset++;
	}

	/*Copy the data from top right section*/
	row_offset = 0;
	row_index = cell_position.y == 0 || cell_position.x == 0 ? cell_position.y : 0;
	for (row_index; row_index < cell_position.y; row_index++)
	{
		memcpy(h_point_buffer + (point_buffer_resolution.x - cell_position.x) * stride_x + row_index * stride_x * point_buffer_resolution.x,
			point_sections[minX][maxY] + row_offset * stride_x * point_buffer_resolution.x,
			sizeof(float) * stride_x * cell_position.x);

		row_offset++;
	}


	/*Send point buffer to the gpu*/
	checkCudaErrors(cudaMemcpy(d_point_buffer, h_point_buffer, sizeof(float) * point_buffer_resolution.x * point_buffer_resolution.y, cudaMemcpyHostToDevice));
}
/*
 * This method sets up a texture object and its respective buffers to share with CUDA device
 *
 * GL_TEXTURE_RECTANGLE is used to avoid generating mip maps and wasting resources
 * Sources: 
 * https://www.khronos.org/opengl/wiki/Rectangle_Texture
 * http://www.songho.ca/opengl/gl_pbo.html
 *
 */
void setupTexture()
{
	// Generate a buffer ID
	glGenBuffers(1, &bufferID);
	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, texture_resolution.x * texture_resolution.y * 3, NULL, GL_DYNAMIC_COPY);
	// Registers the buffer object specified by buffer for access by CUDA.A handle to the registered object is returned as resource.
	// Source: http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OPENGL.html#group__CUDART__OPENGL_1g0fd33bea77ca7b1e69d1619caf44214b
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, bufferID, cudaGraphicsRegisterFlagsNone));

	// Enable Texturing
	glEnable(GL_TEXTURE_RECTANGLE);
	// Generate a texture ID
	glGenTextures(1, &textureID);
	// Make this the current texture (remember that GL is state-based)
	glBindTexture(GL_TEXTURE_RECTANGLE, textureID);
	// Allocate the texture memory. The last parameter is NULL since we only want to allocate memory, not initialize it
	// https://www.khronos.org/opengl/wiki/GLAPI/glTexBuffer
	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB8, texture_resolution.x, texture_resolution.y, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

	// Must set the filter mode to avoid any altering in the resulting image and reduce performance cost
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// Unbind texture and buffer
	glBindTexture(GL_TEXTURE_RECTANGLE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}


//============================
//		OPENGL FUNCTIONS
//============================
/*
 * Update the texture with raytracing data from the GPU
 */
void updateTexture()
{
	//Synchronize OpenGL and CPU calls before locking and working on the buffer object
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));

	//Get a pointer to the memory position in the CUDA device (GPU)
	unsigned char* devPtr;
	size_t size;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&devPtr), &size, cuda_pbo_resource));

	//Call the wrapper function invoking the CUDA Kernel
	CudaSpace::rayTrace(texture_resolution, frame_dimension, camera_forward, camera_position, devPtr, max_height, use_color_map, use_LOD);

	//Synchronize CUDA calls and release the buffer for OpenGL and CPU use;
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}
/*
 * Copy pixel data to a texture and display it on screen
 */
void renderTexture()
{
	int width = glutGet(GLUT_WINDOW_WIDTH),
		height = glutGet(GLUT_WINDOW_HEIGHT);

	// Copy texture data from buffer;
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
	glBindTexture(GL_TEXTURE_RECTANGLE, textureID);
	glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, texture_resolution.x, texture_resolution.y, GL_RGB, GL_UNSIGNED_BYTE, NULL);


	// Adjust coordinate system to screen position
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, width, 0.0, height, -1.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();


	// Draw a quad and apply texture
	glEnable(GL_TEXTURE_RECTANGLE);
	{
		glBegin(GL_QUADS);

		glTexCoord2i(0, 0); 
		glVertex2i(0, 0);

		glTexCoord2i(texture_resolution.x, 0);	
		glVertex2i(width, 0);

		glTexCoord2i(texture_resolution.x, texture_resolution.y);
		glVertex2i(width, height);

		glTexCoord2i(0, texture_resolution.y);
		glVertex2i(0, height);
		glEnd();
	}
	glDisable(GL_TEXTURE_RECTANGLE);

	// Unbind buffer and texture
	glBindTexture(GL_TEXTURE_RECTANGLE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
}


//============================
//		CAMERA FUNCTIONS
//============================

/*
 *Handle the camera movement
 */
void moveCamera()
{
	camera_position += delta_time.count() * (glm::vec3(0, movement_up, 0) + glm::normalize(glm::vec3(camera_forward.x, 0, camera_forward.z)) * movement_fwd + glm::normalize(glm::cross(camera_forward, glm::vec3(0, 1, 0))) * movement_rht);
	
	if (camera_position.x < 0)
		camera_position.x = 0;
	if (camera_position.x >= boundaries.x)
		camera_position.x = boundaries.x - 0.00001f;

	if (camera_position.z < 0)
		camera_position.z = 0;
	if (camera_position.z >= boundaries.y)
		camera_position.z = boundaries.y - 0.00001f;

	if (camera_position.y < 0)
		camera_position.y = 0;
	if (camera_position.y >= max_height * 4)
		camera_position.y = max_height * 4;

}
/*
 * Handle camera rotation
 */
void rotateCamera()
{

	camera_forward = glm::rotate(camera_forward, rotation_up * delta_time.count(), glm::vec3(0, 1, 0));
	camera_forward = glm::rotate(camera_forward, rotation_right * delta_time.count(), glm::normalize(glm::cross(camera_forward, glm::vec3(0, 1, 0))));
}


//============================
//		GLUT FUNCTIONS
//============================
/*Displays FPS from http://stackoverflow.com/questions/20866508/using-glut-to-simply-print-text*/
void drawFPS()
{
	int width = glutGet(GLUT_WINDOW_WIDTH),
		height = glutGet(GLUT_WINDOW_HEIGHT);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, width, 0, height);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glRasterPos2i(10, 10);

	current_frame = sys_clock.now();
	delta_time = current_frame - last_frame;
	last_frame = current_frame;
	std::string text = "FPS " + std::to_string(1 / delta_time.count());
	
	for (int i = 0; i < text.length(); ++i) {
		glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_10, text[i]);
	}

	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
}

/* process menu option 'op' */
void menu(int op)
{
	switch (op)
	{
	case 'Q':
	case 'q':
		exit(0);
	default:;
	}
}

/* executed when a regular key is pressed */
void keyboardDown(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'p':
	case '27':
		exit(0);
	/* Movement */
	case 'e':
		movement_up = qe_movement_distance;
		break;
	case 'q':
		movement_up = -qe_movement_distance;
		break;
	case'a':
		movement_rht = wasd_movement_distance;
		break;
	case'd':
		movement_rht = -wasd_movement_distance;
		break;
	case 'w':
		movement_fwd = wasd_movement_distance;
		break;
	case 's':
		movement_fwd = -wasd_movement_distance;
		break;

	/* Rotation */
	case 'i':
		rotation_right = ik_rotation_angle;
		break;
	case 'k':
		rotation_right = -ik_rotation_angle;
		break;
	case 'j':
		rotation_up = -jl_rotation_angle;
		break;
	case 'l':
		rotation_up = jl_rotation_angle;
		break;

	/* Visualization parameters */
	case 'r':
		if (h_color_map != NULL)
			use_color_map = !use_color_map;
		break;
	case 't':
		use_LOD = !use_LOD;
	default:;
	}
}

void keyboardUp(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'p':
	case '27':
		exit(0);
	case 'e':
	case 'q':
		movement_up = 0;
		break;
	case 'a':
	case 'd':
		movement_rht = 0;
		break;
	case 'w':
	case 's':
		movement_fwd = 0;
		break;
	case 'i':
	case 'k':
		rotation_right = 0;
		break;
	case 'j':
	case 'l':
		rotation_up = 0;
	default:;
	}
}

/* reshaped window */
void reshape(int width, int height)
{
	GLfloat fieldOfView = 90.0f;
	glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fieldOfView, static_cast<GLfloat>(width) / static_cast<GLfloat>(height), 0.1, 500.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

/* executed when button 'button' is put into state 'state' at screen position ('x', 'y') */
void mouseClick(int button, int state, int x, int y)
{
}

/* executed when the mouse moves to position ('x', 'y') */
void mouseMotion(int x, int y)
{
}

/* render the scene */
void draw()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	moveCamera();
	rotateCamera();
	manageSections();

	/* render the scene here */
	copyPointBuffer();
	updateTexture();
	renderTexture();

	drawFPS();

	glFlush();
	glutSwapBuffers();
}

/* executed when program is idle */
void idle()
{
	draw();
}

/* initialize OpenGL settings */
void initGL(int width, int height)
{
	reshape(width, height);

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
}


//============================
//        RESOURCES
//============================
/* Allocate resources */
void initialize()
{
	glewInit();

	checkCudaErrors(cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId()));
	loadJPEG(color_map_file);
	setupTexture();
	h_point_buffer = new float[point_buffer_resolution.x * point_buffer_resolution.y];
	checkCudaErrors(cudaMalloc(&d_point_buffer, sizeof(float) * point_buffer_resolution.x * point_buffer_resolution.y));
	initializeSections();
	CudaSpace::initializeDeviceVariables(point_buffer_resolution, texture_resolution, d_point_buffer, d_color_map, color_map_resolution, cell_size);
}

/* Free Resources */
void freeResourcers()
{
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(d_point_buffer));
	checkCudaErrors(cudaFree(d_color_map));
	CudaSpace::freeDeviceVariables();
	delete[](h_color_map);
	delete[](h_point_buffer);
	for each(std::thread* ptr in thread_pool)
		delete ptr;
	for each(float* ptr in point_sections)
		delete[] ptr;
	
}


//============================
//			MAIN
//============================
/* initialize GLUT settings, register callbacks, enter main loop */
int main(int argc, char** argv)
{
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(1024, 768);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("GPU Heightmap Raytracer 2k17");

	// register glut call backs
	glutKeyboardFunc(keyboardDown);
	glutKeyboardUpFunc(keyboardUp);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMotion);
	glutReshapeFunc(reshape);
	glutDisplayFunc(draw);
	glutIdleFunc(idle);
	glutIgnoreKeyRepeat(true); // ignore keys held down

	// create a sub menu 
	int subMenu = glutCreateMenu(menu);
	glutAddMenuEntry("Do nothing", 0);
	glutAddMenuEntry("Really Quit", 'q');

	// create main "right click" menu
	glutCreateMenu(menu);
	glutAddSubMenu("Sub Menu", subMenu);
	glutAddMenuEntry("Quit", 'q');
	glutAttachMenu(GLUT_RIGHT_BUTTON);


	initialize();
	atexit(freeResourcers);
	initGL(1024, 768);

	glutMainLoop();

	return 0;
}
