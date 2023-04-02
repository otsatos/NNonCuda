# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags)

all: clean build

build: main.cu 
	$(CXX) main.cu activationf.cu loss.cu matrixoperations.cu memallocations.cu feedforwardnn.cu matrix.cu backpropagation.cu training.cu -o test.exe --std c++17  $(CXXFLAGS) -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -lcudart -lcuda -lcublas

run:
	./nn.exe $(ARGS)

clean:
	rm -f nn.exe output*.txt 