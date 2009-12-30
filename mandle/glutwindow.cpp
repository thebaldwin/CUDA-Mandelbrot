#include "glutwindow.h"

int GlutWindow::windowWidth;
int GlutWindow::windowHeight;
float GlutWindow::xshift = 0.0f;
float GlutWindow::yshift = 0.0f;
float GlutWindow::zoomFactor = 1.0f;
int GlutWindow::buttonPressed = GLUT_UP;
int GlutWindow::previousx = 0;
int GlutWindow::previousy = 0;

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
}

void GlutWindow::mouseWheel(int /*wheel*/, int dir, int /*x*/, int /*y*/) {
  if (dir > 0)
    zoomFactor /= 2;
  else
    zoomFactor *= 2;
}

void GlutWindow::mouseFunc(int /*button*/, int state, int x, int y) {
  buttonPressed = state;
  previousx = x;
  previousy = y;
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
}

GLuint GlutWindow::setupPBO(void) {
  GLuint pbo;

  glGenBuffersARB(1, &pbo);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferDataARB(GL_PIXEL_UNPACK_BUFFER, (windowWidth * windowHeight * 4 * sizeof(GLubyte)), NULL, GL_DYNAMIC_DRAW );
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, 0);
  
  return pbo;
}

void GlutWindow::handleKeyInput(unsigned char key, int /*x*/, int /*y*/) {
    switch(key) {
      case(27) :
          exit(0);
          break;
    }
}

void GlutWindow::displayPBO(GLuint pbo) {
  glRasterPos2i(0,0);
    
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, pbo);
  glDrawPixels(windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glBindBufferARB(GL_PIXEL_UNPACK_BUFFER, 0);
}