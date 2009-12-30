#ifndef GLUTWINDOW
#define GLUTWINDOW

#include <GL/glew.h>
#include <GL/freeglut.h>

class GlutWindow {

public:
  static int windowWidth;
  static int windowHeight;

  static float xshift;
  static float yshift;
  static float zoomFactor;

private:
  static int previousx;
  static int previousy;

  static int buttonPressed;

  static const char* DEFAULT_WINDOW_TEXT;

public:
  GlutWindow(const char* windowText = DEFAULT_WINDOW_TEXT, int argc = 0, char** argv = 0, int width=800, int height=600);
  ~GlutWindow();

  GLuint setupPBO();

private:
  static void handleKeyInput(unsigned char, int, int);
  static void setup(const char*, int, char**);

  static void reshape(int, int);
  static void mouseMove(int, int);
  static void mouseFunc(int, int, int, int);
  static void mouseWheel(int, int, int, int);

  static void defaultGlutDrawLoop();
};

#endif