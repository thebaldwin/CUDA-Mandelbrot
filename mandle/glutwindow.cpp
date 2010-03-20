#include "glutwindow.h"
#include <cstdlib>

int GlutWindow::windowWidth;
int GlutWindow::windowHeight;
float GlutWindow::xshift = 0.0f;
float GlutWindow::yshift = 0.0f;
float GlutWindow::zoomFactor = 1.0f;
int GlutWindow::buttonPressed = GLUT_UP;
int GlutWindow::previousx = 0;
int GlutWindow::previousy = 0;
GLuint GlutWindow::texture = 0;

const char* GlutWindow::DEFAULT_WINDOW_TEXT = "Default Window Text";

GlutWindow::GlutWindow(const char* windowText, int argc, char** argv, int width, int height)
{
  windowWidth = width;
  windowHeight = height;

  setup(windowText, argc, argv);
}

GlutWindow::~GlutWindow() {
}

void GlutWindow::setup(const char* windowText, int argc, char** argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

  glutInitWindowSize(windowWidth, windowHeight);

  glutCreateWindow(windowText);
  glewInit();

  glutKeyboardFunc(&GlutWindow::handleKeyInput);
  glutMouseWheelFunc(&GlutWindow::mouseWheel);
  glutMotionFunc(&GlutWindow::mouseMove);
  glutMouseFunc(&GlutWindow::mouseFunc);

  glutReshapeFunc(&GlutWindow::reshape);

  glutDisplayFunc(&GlutWindow::defaultGlutDrawLoop);

  setupTexturing();
}

void GlutWindow::handleKeyInput(unsigned char key, int /*x*/, int /*y*/) {
  switch(key) {
    case(27) :
        exit(0);
        break;
  }
}

void GlutWindow::mouseWheel(int /*wheel*/, int dir, int /*x*/, int /*y*/) {
  if (dir > 0)
    zoomFactor /= 2;
  else
    zoomFactor *= 2;
}

void GlutWindow::mouseMove(int x, int y) {
  if (buttonPressed == GLUT_DOWN) {
    int xPixelChange = -(x - previousx);
    int yPixelChange = -(y - previousy);

    previousx = x;
    previousy = y;

    xshift += zoomFactor*((3.0f / windowWidth) * xPixelChange);
    yshift += zoomFactor*((2.0f / windowHeight) * yPixelChange);
  }
}

void GlutWindow::mouseFunc(int /*button*/, int state, int x, int y) {
  buttonPressed = state;
  previousx = x;
  previousy = y;
}

void GlutWindow::defaultGlutDrawLoop(void) {
  glClearColor(0.0, 1.0, 1.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);

  glutSwapBuffers();
  glutPostRedisplay();
}

void GlutWindow::reshape(int w, int h) {
  windowWidth = w;
  windowHeight = h;

  glViewport(0, 0, GLsizei(w), GLsizei(h));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0.0, GLdouble(w), 0.0, GLdouble(h));
  glMatrixMode(GL_MODELVIEW);

  setupTexturing();
}

GLuint GlutWindow::setupPBO(void) {
  GLuint pbo;

  glGenBuffersARB(1, &pbo);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
  glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, (windowWidth * windowHeight * 4 * sizeof(GLubyte)), 0, GL_STREAM_COPY );
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  
  return pbo;
}
void GlutWindow::setupTexturing() {
  if (texture)
    glDeleteTextures(1, &texture);

  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, windowWidth, windowHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
}

void GlutWindow::displayPBO(GLuint pbo) {
  glRasterPos2i(0,0);
    
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, pbo);
  glDrawPixels(windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, 0);
}

void GlutWindow::displayPBOTexture(GLuint pbo) {
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);

  glDisable(GL_DEPTH_TEST);

  glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f((float)windowWidth, 0.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f((float)windowWidth, (float)windowHeight);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(0.0f, (float)windowHeight);
  glEnd();

  glEnable(GL_DEPTH_TEST);

  glBindTexture(GL_TEXTURE_2D, 0);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, 0);
}
