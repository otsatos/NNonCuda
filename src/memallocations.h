#include <memory>
#include <cuda_runtime.h>

#ifndef HMEMALLOCATIONS
#define HMEMALLOCATIONS

__host__ float *allocateDeviceMemory(int elements);
__host__ float *allocateHostMemory(int elements);
__host__ float *allocateHostMemoryPinned(int elements);
__host__ float *allocateHostMemoryManaged(int elements);

#endif