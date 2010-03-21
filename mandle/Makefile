all:
	/usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -I/usr/include main.cu glutwindow.cpp -lGLEW -lglut -lcudart -L/usr/local/cuda/lib64 -L/usr/local/lib -o fractal
clean:
	rm fractal
