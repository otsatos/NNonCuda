#ifndef HSIGMOID
#define HSIGMOID

#include <iostream>
#include <random>
#include "matrix.h"
#include "memallocations.h"
#include "exceptionext.h"

__device__ float sigmoidScalar(float x);

__global__ void ksigmoidForwardActivation(Matrix Z, Matrix A, int ZxLength, int ZyLength);
__host__ void sigmoidForwardActivation(Matrix Z, Matrix A, int rows, int cols);
Matrix sigmoid(Matrix Z);


__host__ Matrix generateWeights(int rows, int cols);
__host__ float** generateWeights_(int rows, int cols);

void testGenerateWeights(int rows, int cols);

#endif