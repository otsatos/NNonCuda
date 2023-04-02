#include <iostream>
#include <cuda_runtime.h>
#include "exceptionext.h"
#include "memallocations.h"
#include "matrix.h"

__global__ void squareError(Matrix P, Matrix Y, int length, float *sumError);
float mseLoss(Matrix P, Matrix Y);

void testMseLoss();