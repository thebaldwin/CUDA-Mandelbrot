#include "glutwindow.h"

#include <iostream>
#include <ctime>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

void glutDrawLoop(void);
void setWindowTitleStats(void);
void renderMandelbrot(GLuint pbo);
void displayPBO(GLuint pbo);

__global__ void mandel(int width, int height, float xshift, float yshift, float zoomFactor, int* output);

GlutWindow* gw;

int main(int argc, char** argv) {
  gw = new GlutWindow("Mandelbrot");
  glutDisplayFunc(glutDrawLoop);

  glutMainLoop();
}

void glutDrawLoop(void) {
  glClearColor(0.0f, 1.0f, 1.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  GLuint pbo = gw->setupPBO();
  renderMandelbrot(pbo);
  gw->displayPBO(pbo);

  setWindowTitleStats();

  glutSwapBuffers();
  glutPostRedisplay();

  glDeleteBuffers(1, &pbo);
}

void setWindowTitleStats(void) {
  static int t1 = 0;
  int t2diff = clock() - t1;
  
  float fps = 1.0f / t2diff * 1000;
  
  const int TITLE_SIZE = 200;

  char title[TITLE_SIZE];
  memset(title, 0, TITLE_SIZE);
  sprintf_s(title, TITLE_SIZE, "zoomFactor: %f, FPS: %f", gw->zoomFactor, fps);

  glutSetWindowTitle(title);
  t1 = clock();
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
  const int ITERATIONS = 100;

  int pixelx = blockIdx.x * blockDim.x + threadIdx.x;
  int pixely = blockIdx.y * blockDim.y + threadIdx.y;

  if ((pixelx >= width) || (pixely >= height))
    return;

  int pixelLoc = (width * pixely) + pixelx;

  float xstart = (-2.0f * zoomFactor) + xshift;
  float ystart = (1.0f * zoomFactor) + yshift;

  float widthPerPixel = (3.0f * zoomFactor) / width;
  float heightPerPixel = (2.0f * zoomFactor) / height;

  float2 zn, c;
  c.x = xstart + (widthPerPixel * pixelx);
  c.y = ystart - (heightPerPixel * pixely);

  zn.x = 0.0f;
  zn.y = 0.0f;
  
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
        r = (int)((i / halfMaxIterations) * 255);
        g = 0;
        b = 0;
      } else {
        float brightness = ((i - halfMaxIterations) / halfMaxIterations);

        r = 255;
        g = (int)(brightness * 255) << 8;
        b = (int)(brightness * 255) << 16;
      }

      output[pixelLoc] = (r | g | b);
      break;
    }
  }
}

