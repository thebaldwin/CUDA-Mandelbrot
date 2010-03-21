#include "glutwindow.h"

#include <iostream>
#include <ctime>

#ifdef _MSC_VER
  #define snprintf _snprintf
#endif

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

void glutDrawLoop(void);
void setWindowTitleStats(void);
void renderMandelbrot(GLuint pbo);
void displayPBO(GLuint pbo);

__global__ void mandel(int width, int height, float xshift, float yshift, float zoomFactor, int* output);
__device__ int calcColour(float2 zn, float2 c);

GlutWindow* gw;

int main(int argc, char** argv) {
  gw = new GlutWindow("Mandelbrot");
  glutDisplayFunc(glutDrawLoop);

  glutMainLoop();
}

void glutDrawLoop(void) {
  glClearColor(0.8f, 1.0f, 1.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glClearDepth(0.0f);

  glLoadIdentity();

  GLuint pbo = gw->setupPBO();
  renderMandelbrot(pbo);
  gw->displayPBOTexture(pbo);
  
  setWindowTitleStats();

  glutSwapBuffers();
  glutPostRedisplay();

  glDeleteBuffers(1, &pbo);
}

void setWindowTitleStats(void) {
  static timespec t1;
  timespec tnow;
  clock_gettime(CLOCK_REALTIME, &tnow);
          
  long t2diff = tnow.tv_nsec - t1.tv_nsec;
  
  float fps = 1.0f / t2diff * 1000000000;
  
  const int TITLE_SIZE = 200;

  char title[TITLE_SIZE];
  memset(title, 0, TITLE_SIZE);
  snprintf(title, TITLE_SIZE, "zoomFactor: %f, FPS: %f", gw->zoomFactor, fps);

  glutSetWindowTitle(title);
  clock_gettime(CLOCK_REALTIME, &t1);
}

void renderMandelbrot(GLuint pbo) {
  cudaGLRegisterBufferObject(pbo);

  int *cudaPBO;
  cudaGLMapBufferObject((void**)&cudaPBO, pbo);

  dim3 dimBlock(16, 16);
  dim3 dimGrid((gw->windowWidth + dimBlock.x - 1) / dimBlock.x, (gw->windowHeight + dimBlock.y - 1) / dimBlock.y);

  mandel<<<dimGrid, dimBlock>>>(gw->windowWidth, gw->windowHeight, gw->xshift, gw->yshift, gw->zoomFactor, cudaPBO);

  cudaThreadSynchronize();

  cudaGLUnmapBufferObject(pbo);
  cudaGLUnregisterBufferObject(pbo);
}

__global__ void mandel(int width, int height, float xshift, float yshift, float zoomFactor, int* output) {
  int2 pixel = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                         blockIdx.y * blockDim.y + threadIdx.y);

  if ((pixel.x >= width) || (pixel.y >= height))
    return;

  int pixelLoc = (width * pixel.y) + pixel.x;

  float2 start = make_float2((-2.0f * zoomFactor) + xshift,
                             (1.0f * zoomFactor) + yshift);

  float widthPerPixel = (3.0f * zoomFactor) / width;
  float heightPerPixel = (2.0f * zoomFactor) / height;

  float2 c = make_float2(start.x + (widthPerPixel * pixel.x),
                          start.y - (heightPerPixel * pixel.y));
  
  float2 zn = make_float2(0.0f, 0.0f);
  //float2 zn = make_float2(-0.1f, 0.7f); //make_float2(0.39f, -0.2f);
  
  output[pixelLoc] = calcColour(zn, c);
}

__device__ int calcColour(float2 zn, float2 c) {
  const int ITERATIONS = 100;

  for(int i = 0; i < ITERATIONS; i++) {

    //zn = zn*zn + c
    float newznx = (zn.x * zn.x) - (zn.y * zn.y) + c.x;
    float newzny = (zn.y * zn.x) + (zn.x * zn.y) + c.y;
    
    zn.x = newznx;
    zn.y = newzny;

    if ((zn.x * zn.x) + (zn.y * zn.y) > 4) {
      int r=0,g=0,b=0;

      //colour it white if it took longer than half the max iterations leave the set
      float halfMaxIterations = ITERATIONS/2.0f;

      if (i < halfMaxIterations) {
        r = 0;
        g = (int)((i / halfMaxIterations) * 255);
        b = (int)((i / halfMaxIterations) * 255);
      } else {
        float brightness = ((i - halfMaxIterations) / halfMaxIterations);

        r = 255;
        g = (int)(brightness * 255);
        b = (int)(brightness * 255);
      }

      return (r | (g << 8) | (b << 16) );
    }
  }
  return 0;
}

