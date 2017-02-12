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

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>

#include "jpeglib.h"

#include <liblas/liblas.hpp>
#include <fstream>
#include <iostream> 

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "CudaKernel.cuh"
#include "helper_cuda.h"


//============================
//		GLOBAL VARIABLES
//============================

std::string point_cloud_file = "autzen.las";
std::string color_map_file = "autzen.jpg";

glm::ivec2 texture_resolution(640, 480);
glm::vec3
	camera_position(0, 800, 0),
	camera_forward = glm::normalize(glm::vec3(0, -.9, .1)),
	frame_dimension(40, 30, 30); //width, height, distance from camera

GLuint textureID;
GLuint bufferID;

// JPEG image
unsigned char *h_color_map = NULL;
glm::ivec2 color_map_resolution = glm::zero<glm::ivec2>();

// Point buffer to be copied to GPU
float* h_point_buffer;
glm::ivec2 const point_buffer_res = glm::ivec2(1024, 1024);

// CPU-Side point grid
float* cpu_point_grid;
glm::ivec2 cpu_point_grid_resolution = glm::zero<glm::ivec2>();
glm::vec2 const cell_size = glm::vec2(1, 1);

// clock
std::chrono::system_clock sys_clock;
std::chrono::time_point<std::chrono::system_clock> last_frame, current_frame;
std::chrono::duration<float> delta_time;

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

	unsigned char *image_buffer, *it;
	unsigned char r, g, b;
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	int width, height;
	JSAMPARRAY pJpegBuffer;

	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);

	/*Open JPEG and start decompression*/
	int row_stride;
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
	image_buffer = new unsigned char[width*height * 3];
	it = image_buffer + width * height * 3;

	row_stride = width * cinfo.output_components;
	pJpegBuffer = (*cinfo.mem->alloc_sarray)
		(reinterpret_cast<j_common_ptr>(&cinfo), JPOOL_IMAGE, row_stride, 1);

	/*Read data in JPEG - scan from top to bottom*/
	while (cinfo.output_scanline < cinfo.output_height)
	{
		jpeg_read_scanlines(&cinfo, pJpegBuffer, 1);
		it -= row_stride;
		for (int x = 0; x < width; x++)
		{
			r = pJpegBuffer[0][cinfo.output_components * x];
			g = pJpegBuffer[0][cinfo.output_components * x + 1];
			b = pJpegBuffer[0][cinfo.output_components * x + 2];

			*(it++) = r;
			*(it++) = g;
			*(it++) = b;
		}
		it -= row_stride;
	}

	/*Finish decompression and release object*/
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(infile);

	h_color_map = image_buffer;
	color_map_resolution = glm::ivec2(width, height);

	/*Pass values to GPU*/
	size_t count = sizeof(unsigned char) * 3 * width * height;
	checkCudaErrors(cudaMalloc(&d_color_map, count));
	checkCudaErrors(cudaMemcpy(d_color_map, h_color_map, count, cudaMemcpyHostToDevice));
	
	return true;
}

/*
 * Load points using libLas Library in LAS format
 * Snippet from: http://www.liblas.org/tutorial/cpp.html
 */
void loadPointDataLAS(std::string filename)
{
	/*Create input stream and associate it with .las file opened to read in binary mode*/
	std::ifstream ifs;
	ifs.open("../Data/" + filename, std::ios::in | std::ios::binary);
	if(!ifs.is_open())
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
	float deltaX, deltaY;
	deltaX = header.GetMaxX() - header.GetMinX();
	deltaY = header.GetMaxY() - header.GetMinY();
	std::cout << "DiffX: " << deltaX << " DiffY: " << deltaY << std::endl;

	cpu_point_grid_resolution = glm::ivec2(glm::floor(deltaX), glm::floor(deltaY));
	cpu_point_grid = new float[cpu_point_grid_resolution.x * cpu_point_grid_resolution.y];

	/*Iterate through point records and calculate the height contribution to each neighboring grid cell*/
	while (reader.ReadNextPoint())
	{
		liblas::Point const& p = reader.GetPoint();
		int x, y, index;
		float alpha, fX, fY, fZ;

		fX = p.GetX() - header.GetOffsetX();
		fY = p.GetY() - header.GetOffsetY();
		fZ = p.GetZ() - header.GetOffsetZ();
		x = static_cast<int>(glm::floor(fX));
		y = static_cast<int>(glm::floor(fY));

		if(x >= 0 && x < cpu_point_grid_resolution.x)
		{
			//bottom left
			if (y >= 0 && y < cpu_point_grid_resolution.y)
			{
				index = x + y * cpu_point_grid_resolution.x;
				alpha = glm::sqrt(glm::pow(fX - x, 2) + glm::pow(fY - y, 2));
				cpu_point_grid[index] = cpu_point_grid[index] * (1 - alpha) + fZ * alpha;
			}

			//top left
			if (y + 1 >= 0 && y + 1 < cpu_point_grid_resolution.y)
			{
				index = x + (y + 1)*cpu_point_grid_resolution.x;
				alpha = glm::sqrt(glm::pow(fX - x, 2) + glm::pow(cell_size.y - (fY - y), 2));
				cpu_point_grid[index] = cpu_point_grid[index] * (1 - alpha) + fZ * alpha;
			}
		}

		if (x + 1 >= 0 && x + 1 < cpu_point_grid_resolution.x)
		{
			//bottom right
			if (y >= 0 && y < cpu_point_grid_resolution.y)
			{
				index = x + 1 + y * cpu_point_grid_resolution.x;
				alpha = glm::sqrt(glm::pow(cell_size.x - (fX - x), 2) + glm::pow(fY - y, 2));
				cpu_point_grid[index] = cpu_point_grid[index] * (1 - alpha) + fZ * alpha;
			}

			//top right
			if (y + 1 >= 0 && y + 1 < cpu_point_grid_resolution.y)
			{
				index = x + 1 + (y + 1) * cpu_point_grid_resolution.x;
				alpha = glm::sqrt(glm::pow(cell_size.x - (fX - x), 2) + glm::pow(cell_size.y - (fY - y), 2));
				cpu_point_grid[index] = cpu_point_grid[index] * (1 - alpha) + fZ * alpha;
			}
		}

	}

	/*Close the file stream*/
	ifs.close();
}

/*Load Point Data from disk -- TESTING PURPOSE ONLY*/
void loadPointDataXYZ(std::string filename)
{
	std::cout << "Loading points... ";

	std::ifstream file("../Data/" + filename);
	std::string line, data;

	cpu_point_grid_resolution = glm::ivec2(1025, 1025);
	cpu_point_grid = new float[cpu_point_grid_resolution.x * cpu_point_grid_resolution.y];

	float x, y, z;
	while (std::getline(file, line) && !line.empty())
	{
		size_t start = 0, end = line.find(" ");
		data = line.substr(start, end - start);
		x = (stof(data));

		start = end + 1;
		end = line.find(" ", start + 1);
		data = line.substr(start, end - start);
		z = (stof(data));

		start = end + 1;
		data = line.substr(start, end - start);
		y = (stof(data));

		cpu_point_grid[int(glm::floor(x)) + cpu_point_grid_resolution.y * int(glm::floor(z))] = y;
	}
	file.close();

	std::cout << "done" << std::endl;
}

/*Snippet from http://www.nvidia.com/content/GTC/documents/1055_GTC09.pdf
 * and http://www.songho.ca/opengl/gl_pbo.html
 *
 * This method sets up a texture object and its respective buffers to share with CUDA device
 *
 * GL_TEXTURE_RECTANGLE is used to avoid generating mip maps and wasting resources
 * Source: https://www.khronos.org/opengl/wiki/Rectangle_Texture
 *
 */
 // Setup Texture
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

	//Call the wrapper method invoking the CUDA Kernel

	CudaSpace::rayTrace(texture_resolution, frame_dimension, camera_forward, camera_position.y, devPtr);

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
//		THREAD FUNCTIONS
//============================
/*
 * Load point data closest to the camera from the HDD
 */
void loadPointsFromDisk()
{
	//TODO: implement
}

/*
 * Set up CPU-Side buffer that is going to be transferred over to the GPU
 */
void setUpGPUPointBuffer()
{
	//TODO: implement

	//Send gpu point buffer to the gpu
	checkCudaErrors(cudaMemcpy(d_point_buffer, cpu_point_grid, sizeof(float) * cpu_point_grid_resolution.x * cpu_point_grid_resolution.y, cudaMemcpyHostToDevice));
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
	case 'Q':
	case 'q':
	case 27:
		// ESC
		exit(0);
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

	/* render the scene here */
	setUpGPUPointBuffer();
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
	loadPointDataLAS(point_cloud_file);
	setupTexture();

	checkCudaErrors(cudaMalloc(&d_point_buffer, sizeof(float) * cpu_point_grid_resolution.x * cpu_point_grid_resolution.y));
	h_point_buffer = new float[cpu_point_grid_resolution.x * cpu_point_grid_resolution.y];
	CudaSpace::initializeDeviceVariables(cpu_point_grid_resolution, texture_resolution, d_point_buffer, d_color_map, color_map_resolution);

}

/* Free Resources */
void freeResourcers()
{
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(d_point_buffer));
	checkCudaErrors(cudaFree(d_color_map));
	delete[](h_point_buffer);
	delete[](h_color_map);
	delete[](cpu_point_grid);
	CudaSpace::freeDeviceVariables();
}


//============================
//			MAIN
//============================
/* initialize GLUT settings, register callbacks, enter main loop */
int main(int argc, char** argv)
{
	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(800, 600);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("GPU Heightmap Raytracer 2k17");

	// register glut call backs
	glutKeyboardFunc(keyboardDown);
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
	initGL(800, 600);

	glutMainLoop();

	return 0;
}
