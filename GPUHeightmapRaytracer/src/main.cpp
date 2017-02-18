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
glm::ivec2 texture_resolution(640, 480);
glm::vec3
	camera_position(1500, 200, 0),
	camera_forward = glm::normalize(glm::vec3(5, -.9, 0)),
	frame_dimension(40, 30, 30); //width, height, distance from camera

GLuint textureID;
GLuint bufferID;

// JPEG image
unsigned char *h_color_map = NULL;
glm::ivec2 color_map_resolution = glm::zero<glm::ivec2>();

// Point buffer to be copied to GPU
float* h_point_buffer;
glm::ivec2 point_buffer_resolution = glm::ivec2(1024 * 5, 1024 * 5);

// CPU-Side point grid
float* cpu_point_grid;
glm::ivec2 cpu_point_grid_resolution = glm::zero<glm::ivec2>();
glm::vec2 cell_size;
float max_height = 0;
float height_tolerance = 50;

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
	double deltaX, deltaY;
	deltaX = header.GetMaxX() - header.GetMinX();
	deltaY = header.GetMaxY() - header.GetMinY();
	std::cout << "DiffX: " << deltaX << " DiffY: " << deltaY << std::endl;

	/*Calculate area per point to set cell dimension*/
	float point_area = float(deltaX * deltaY / header.GetPointRecordsCount());
	cell_size.x = point_area;
	cell_size.y = point_area;

	/*Allocate Grids*/
	cpu_point_grid_resolution = glm::ivec2(glm::floor(deltaX / cell_size.x) + 1, glm::floor(deltaY / cell_size.y) + 1);
	cpu_point_grid = new float[cpu_point_grid_resolution.x * cpu_point_grid_resolution.y]();
	std::cout << "Grid Resolution: " << cpu_point_grid_resolution.x << " " <<cpu_point_grid_resolution.y << std::endl;

	/*Reduce size of gpu grid if the cpu grid is too small*/
	if (point_buffer_resolution.x > cpu_point_grid_resolution.x)
		point_buffer_resolution.x = cpu_point_grid_resolution.x;
	if (point_buffer_resolution.y > cpu_point_grid_resolution.y)
		point_buffer_resolution.y = cpu_point_grid_resolution.y;

	/*Iterate through point records and calculate the height contribution to each neighboring grid cell*/
	while (reader.ReadNextPoint())
	{
		liblas::Point const& p = reader.GetPoint();
		int x, y, index;
		float fX, fY, fZ;

		fX = float(p.GetX() - header.GetMinX());
		fY = float(p.GetY() - header.GetMinY());
		fZ = float(p.GetZ() - header.GetMinZ());
		x = static_cast<int>(glm::floor(fX / cell_size.x));
		y = static_cast<int>(glm::floor(fY / cell_size.y));

		index = x + y * cpu_point_grid_resolution.x;
		cpu_point_grid[index] = fZ;

		if (max_height < fZ && fZ - max_height < height_tolerance)
		{
			max_height = fZ;
		}
	}

	/*Close the file stream*/
	ifs.close();
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

	//Call the wrapper function invoking the CUDA Kernel
	CudaSpace::rayTrace(texture_resolution, frame_dimension, camera_forward, camera_position, devPtr, max_height);

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

/*
* Set up CPU-Side buffer that is going to be transferred over to the GPU
* 
* NOTE: it is more efficient to pre-allocate the point
* buffer in the RAM and then pass it to the GPU
* than passing every line at a time 
*/
void copyPointBuffer()
{
	/*Copies a portion of the grid to raytrace in the gpu*/
	int x, y;
	x = int(floor(camera_position.x - point_buffer_resolution.x / 2 * cell_size.x)) ;
	y = int(floor(camera_position.z - point_buffer_resolution.y / 2 * cell_size.y)) ;
	int size = sizeof(float) * point_buffer_resolution.x;
	for (int i = 0; i < point_buffer_resolution.y; i++)
	{
		int offset = x + (y + i) * cpu_point_grid_resolution.x;
		memcpy(h_point_buffer + i * point_buffer_resolution.x, cpu_point_grid + offset, size);
	}

	/*Send gpu point buffer to the gpu*/
	checkCudaErrors(cudaMemcpy(d_point_buffer, h_point_buffer, sizeof(float) * point_buffer_resolution.x * point_buffer_resolution.y, cudaMemcpyHostToDevice));
}

/*
 *Handle the camera movement
 */
void moveCamera()
{
	camera_position += delta_time.count() * (glm::vec3(0, movement_up, 0) + glm::normalize(glm::vec3(camera_forward.x, 0, camera_forward.z)) * movement_fwd + glm::normalize(glm::cross(camera_forward, glm::vec3(0, 1, 0))) * movement_rht);
	
	if (camera_position.x < (point_buffer_resolution.x / 2) * cell_size.x)
		camera_position.x = (point_buffer_resolution.x / 2) * cell_size.x;
	if (camera_position.x >= (cpu_point_grid_resolution.x - point_buffer_resolution.x / 2) * cell_size.x)
		camera_position.x =  (cpu_point_grid_resolution.x - point_buffer_resolution.x / 2) * cell_size.x;

	if (camera_position.z < (point_buffer_resolution.y / 2) * cell_size.y)
		camera_position.z = (point_buffer_resolution.y / 2) * cell_size.y ;
	if (camera_position.z >= (cpu_point_grid_resolution.y - point_buffer_resolution.y / 2) * cell_size.y)
		camera_position.z =  (cpu_point_grid_resolution.y - point_buffer_resolution.y / 2) * cell_size.y;

	if (camera_position.y < 0)
		camera_position.y = 0;
	if (camera_position.y >= max_height * 2)
		camera_position.y = max_height * 2;

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
//		THREAD FUNCTIONS
//============================
/*
 * Load point data closest to the camera from the HDD
 */
void loadPointsFromDisk()
{
	//TODO: implement
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
	/*movement*/
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

	/*rotation*/
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
	loadPointDataLAS(point_cloud_file);
	setupTexture();
	h_point_buffer = new float[point_buffer_resolution.x * point_buffer_resolution.y];
	checkCudaErrors(cudaMalloc(&d_point_buffer, sizeof(float) * point_buffer_resolution.x * point_buffer_resolution.y));
	CudaSpace::initializeDeviceVariables(point_buffer_resolution, texture_resolution, d_point_buffer, d_color_map, color_map_resolution, cell_size);
}

/* Free Resources */
void freeResourcers()
{
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(d_point_buffer));
	checkCudaErrors(cudaFree(d_color_map));
	delete[](h_color_map);
	delete[](cpu_point_grid);
	delete[](h_point_buffer);
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
	initGL(800, 600);

	glutMainLoop();

	return 0;
}
