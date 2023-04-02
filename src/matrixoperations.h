#ifndef HMATRIXOPS
#define HMATRIXOPS

#include<iostream>
#include<cuda_runtime.h>
#include<cublas_v2.h>
#include "memallocations.h"
#include "matrix.h"

using namespace std;

__global__ void kdotProduct(Matrix A,Matrix B, Matrix C);
__host__ void dotProduct(Matrix A, Matrix B, Matrix C);
void dotProductCublas(Matrix A, Matrix B, Matrix C);

__global__ void kmatrixTranspose(Matrix A, Matrix At);
__host__ void matrixTranspose(float *input, float *output, int width, int height);
void matrixTransposeCublas(Matrix A, Matrix At);

__global__ void kmatrixMultiplyElWise(Matrix A, Matrix B, Matrix C);
__host__ void matrixMultiplyElWise(Matrix A, Matrix B, Matrix C);

void testMAtrixTranspose();
void testmatrixMultiplyElWise();

void testDotProduct();

#endif