
#include<cuda_runtime.h>
#include "loss.h"

using namespace std;


__global__ void squareError(Matrix P, Matrix Y, int length,float *sumError)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < length)
    {
        float e=P.elements[i] - Y.elements[i];        
        atomicAdd(sumError,e * e);       
    } 
}

__host__ float mseLossCaller(Matrix P,Matrix Y)
{
    int length = P.width * P.height;
    dim3 threadsPerBlock(256,1);
	dim3 blocksPerGrid((length + threadsPerBlock.x - 1) / threadsPerBlock.x,1);
        

    float *dsumError;
    cudaMallocManaged(&dsumError, sizeof(float));   
    *dsumError=0.0f;            

	squareError<<<blocksPerGrid, threadsPerBlock>>>(P, Y,length,dsumError);	    
    cudaDeviceSynchronize();    
    //ExceptionExt::throwCudaError("An error occured during kernel execution!");    

    float hsumerror= *dsumError;        
    float lossError = hsumerror/((float)length);    
    
    //*dsumError=0.0f;
    cudaFree(dsumError); 
    return lossError;
}

float mseLoss(Matrix P,Matrix Y)
{
    int n=P.width * P.height;
        
    Matrix Pd ={P.width,P.height,allocateDeviceMemory(n)};
    Matrix Yd ={Y.width,Y.height,allocateDeviceMemory(n)};    
    
	cudaMemcpy(Pd.elements, P.elements, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Yd.elements, Y.elements, n*sizeof(float), cudaMemcpyHostToDevice);
        
    auto mse = mseLossCaller(Pd,Yd);
    return mse;   
}

