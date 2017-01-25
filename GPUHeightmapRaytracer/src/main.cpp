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

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "CudaKernel.cuh"
#include "helper_cuda.h"


#define ar(r, c) (ar[ r*gpu_grid_res + c])

//============================
//		GLOBAL VARIABLES
//============================

int const gpu_grid_res = 1024;
int const cpu_grid_res = 2048;

glm::ivec2 const window_resolution(800, 600);

float cpu_pointBuffer[cpu_grid_res][cpu_grid_res];

GLuint textureID;
GLuint bufferID;

//============================
//		CUDA VARIABLES
//============================

float *d_gpu_pointBuffer;
float *h_gpu_pointBuffer;

struct cudaGraphicsResource *cuda_pbo_resource;

//============================
//		HELPER FUNCTIONS
//============================

/*Load Point Data from disk -- TESTING PURPOSE ONLY*/
void loadPointData()
{
	std::cout << "Loading points... ";

	std::ifstream file("../Data/data");
	std::string line, data;

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

		cpu_pointBuffer[int(x)][int(z)] = y;
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
	glBufferData(GL_PIXEL_UNPACK_BUFFER, window_resolution.x * window_resolution.y * 4, NULL, GL_DYNAMIC_COPY);
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
	glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA8, window_resolution.x, window_resolution.y, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	// Must set the filter mode to avoid any altering in the resulting image and reduce performance cost
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP);

	// Unbind texture and buffer
	glBindTexture(GL_TEXTURE_RECTANGLE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void updateTexture()
{
	//Synchronize OpenGL and CPU calls before locking and working on the buffer object
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));

	//Get a pointer to the memory position in the CUDA device (GPU)
	unsigned char *devPtr;
	size_t size;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&devPtr), &size, cuda_pbo_resource));

	//Call the wrapper method invoking the CUDA Kernel
	//TODO: optmize Grid and Block sizes
	dim3 gridSize = window_resolution.x;
	dim3 blockSize = window_resolution.y;
	CudaSpace::rayTrace_w(gridSize, blockSize, devPtr, d_gpu_pointBuffer);

	//Synchronize CUDA calls and release the buffer for OpenGL and CPU use;
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

void renderTexture()
{

	int width = glutGet(GLUT_WINDOW_WIDTH),
		height = glutGet(GLUT_WINDOW_HEIGHT);

	// Copy texture data from buffer;
	glBindTexture(GL_TEXTURE_RECTANGLE, textureID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
	glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, window_resolution.x, window_resolution.y, GL_BGRA, GL_UNSIGNED_BYTE, NULL);


	//TODO: why??
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
			glTexCoord2i(0, 0); glVertex2i(0, 0);
			glTexCoord2i(1, 0); glVertex2i(width, 0);
			glTexCoord2i(1, 1); glVertex2i(width, height);
			glTexCoord2i(0, 1); glVertex2i(0, height);
		glEnd();
	}
	glDisable(GL_TEXTURE_RECTANGLE);

	// Unbind buffer and texture
	glBindTexture(GL_TEXTURE_RECTANGLE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	//TODO: why??
	glDisable(GL_TEXTURE_2D);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
}

//============================
//		THREAD FUNCTIONS
//============================


//============================
//		GLUT FUNCTIONS
//============================

/* process menu option 'op' */
void menu(int op) {

	switch (op) {
	case 'Q':
	case 'q':
		exit(0);
	}
}

/* executed when a regular key is pressed */
void keyboardDown(unsigned char key, int x, int y) {

	switch (key) {
	case 'Q':
	case 'q':
	case 27:
		// ESC
		exit(0);
	}
}

/* reshaped window */
void reshape(int width, int height) {

	GLfloat fieldOfView = 90.0f;
	glViewport(0, 0, (GLsizei)width, (GLsizei)height);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fieldOfView, (GLfloat)width / (GLfloat)height, 0.1, 500.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

/* executed when button 'button' is put into state 'state' at screen position ('x', 'y') */
void mouseClick(int button, int state, int x, int y) {

}

/* executed when the mouse moves to position ('x', 'y') */
void mouseMotion(int x, int y) {

}

/* render the scene */
void draw() {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	/* render the scene here */
	updateTexture();
	renderTexture();

	glFlush();
	glutSwapBuffers();
}

/* executed when program is idle */
void idle() {
	draw();
}

/* initialize OpenGL settings */
void initGL(int width, int height) {

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
	checkCudaErrors(cudaMalloc(&d_gpu_pointBuffer, sizeof(float) * gpu_grid_res * gpu_grid_res));
	h_gpu_pointBuffer = new float[gpu_grid_res*gpu_grid_res];

	loadPointData();
	setupTexture();
}

/* Free Resources */
void freeResourcers()
{
	checkCudaErrors(cudaFree(d_gpu_pointBuffer));
	free(h_gpu_pointBuffer);
}


//============================
//			MAIN
//============================
/* initialize GLUT settings, register callbacks, enter main loop */
int main(int argc, char** argv) {

	glutInit(&argc, argv);

	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(window_resolution.x, window_resolution.y);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("GPU Heightmap Raytracer 2k16");

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
	initGL(window_resolution.x, window_resolution.y);

	glutMainLoop();

	return 0;
}
