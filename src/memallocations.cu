
#include "memallocations.h"

__host__ float* allocateDeviceMemory(int elements)
{
    size_t size = elements * sizeof(float); 
      
    float *d;    
    cudaMalloc(&d, size);    
    
    return d;
}

__host__ float* allocateHostMemory(int elements)
{
    size_t size = elements * sizeof(float);  
    float *h = (float *)malloc(size);
    return h;
}

__host__ float* allocateHostMemoryPinned(int elements)
{
    size_t size = elements * sizeof(float);  
    float *h;
    cudaMallocHost((float **)&h, size);
    return h;
}

__host__ float* allocateHostMemoryManaged(int elements)
{
    size_t size = elements * sizeof(float);  
    float *h;
    cudaMallocManaged((float **)&h, size);
    return h;
}