#include <iostream>
#include <complex>
#include <GL/glew.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <windows.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace std;

void handleKeyInput(unsigned char, int, int);
void setup(int, char**);

GLuint setupPBO();
void generateMandelbrot(GLuint pbo);
void drawMandelbrot(GLuint pbo);

void display(void);
void reshape(int, int);
void countFPS(void);

int WIDTH = 512;
int HEIGHT = 400;

__global__ void mandel(int width, int height, int* output) {

  int pixelx = blockIdx.x * blockDim.x + threadIdx.x;
  int pixely = blockIdx.y * blockDim.y + threadIdx.y;

  if (pixelx > width || pixely > height)
    return;

  float xstart = -2.0f;
  float ystart = 1.0f;

  float widthPerPixel = 3.0f / width;
  float heightPerPixel = 2.0f / height;

  float2 c;
  c.x = xstart + widthPerPixel * pixelx;
  c.y = ystart - heightPerPixel * pixely;

  float2 zn;
  zn.x = 0;
  zn.y = 0;
  
  int pixelLoc = width*pixely + pixelx;

  for(int i = 0;i<50;i++) {
    //zn = zn*zn + c
    float newznx = zn.x * zn.x - zn.y*zn.y + c.x;
    float newzny = zn.y * zn.x + zn.x * zn.y + c.y;
    
    zn.x = newznx;
    zn.y = newzny;
  
    if (zn.x*zn.x + zn.y*zn.y > 4) {
      int r,g,b;

      //colour it white if it took longer to leave the set
      if (i < 50/2) {
        r = (int)((i/25.0f)*255);
        g = 0;
        b = 0;
      }
      else {
        r = 255;
        g = (int)(((i-25)/25.0f)*255) << 8;
        b = (int)(((i-25)/25.0f)*255) << 16;
      }

      output[pixelLoc] = r | g | b ;
      break;
    }
  }
}

int main(int argc, char** argv) {
  setup(argc, argv);
  glutMainLoop();
}

void setup(int argc, char** argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(WIDTH, HEIGHT);
  glutCreateWindow("Mandelbrot");
  glutKeyboardFunc(handleKeyInput);
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  
  glewInit();
}

void reshape(int w, int h) {
  WIDTH = w;
  HEIGHT = h;

  glViewport(0,0,GLsizei(w),GLsizei(h));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0,GLdouble(w),0.0,GLdouble(h));
  glMatrixMode(GL_MODELVIEW);
}

void display(void) {
  glClearColor(0.0, 1.0, 1.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);

  GLuint pbo = setupPBO();
  generateMandelbrot(pbo);
  drawMandelbrot(pbo);

  countFPS();

  glutSwapBuffers();
  glutPostRedisplay();

  glDeleteBuffers(1, &pbo);
}

void drawMandelbrot(GLuint pbo) {
  glRasterPos2i(0,0);
    
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, pbo);
  glDrawPixels(WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, 0);
}

GLuint setupPBO(void) {
  GLuint pbo;

  glGenBuffersARB(1, &pbo);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferDataARB(GL_PIXEL_UNPACK_BUFFER, WIDTH*HEIGHT*4*sizeof(GLubyte), NULL, GL_DYNAMIC_DRAW );
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, 0);

  return pbo;
}
void countFPS(void) {
  static int t1 = 0;
  int t2diff = timeGetTime() - t1;
  
  float fps = 1.0f / t2diff * 1000;

  char title[200];
  memset(title,0,200);
  sprintf_s(title, 200, "FPS: %f", fps);

  glutSetWindowTitle(title);
  t1 = timeGetTime();
}

void generateMandelbrot(GLuint pbo) {
  cudaGLRegisterBufferObject(pbo);

  int *p;
  cudaGLMapBufferObject((void**)&p, pbo);

  dim3 dimBlock(16, 16);
  dim3 dimGrid((WIDTH + dimBlock.x - 1) / dimBlock.x, (HEIGHT + dimBlock.y - 1) / dimBlock.y);

  mandel<<<dimGrid, dimBlock>>>(WIDTH, HEIGHT, p);

  cudaGLUnmapBufferObject(pbo);
  cudaGLUnregisterBufferObject(pbo);
}

void handleKeyInput(unsigned char key, int x, int y) {
    switch(key) {
    case(27) :
        exit(0);
        break;
    }
}
