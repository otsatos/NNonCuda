#include "activationf.h"


__device__ float sigmoidScalar(float x) 
{
    return 1.0f / (1 + exp(-x));
}

__global__ void ksigmoidForwardActivation(Matrix Z, Matrix A,int Zwidth, int Zheight) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < Zwidth * Zheight) 
    {
	    A.elements[i] = sigmoidScalar(Z.elements[i]);     
    }
}

__host__ void sigmoidForwardActivation(Matrix Z,Matrix A)
{          
    dim3 blockSize(256);        
    dim3 numOfBlocks((Z.height * Z.width + blockSize.x - 1) / blockSize.x);

    ksigmoidForwardActivation<<<numOfBlocks, blockSize>>>(Z, A,Z.width,Z.height);	
    //throwIfCudaErrorOccured__("sigmoidForwardActivation");
}

Matrix sigmoid(Matrix Z)
{
    Matrix A ={Z.width,Z.height,allocateHostMemory(Z.width * Z.height)};

    Matrix Zd={Z.width,Z.height,allocateDeviceMemory(Z.width * Z.height)};  
    Matrix Ad={Z.width,Z.height,allocateDeviceMemory(Z.width * Z.height)};  
    
    cudaMemcpy(Zd.elements, Z.elements, Z.width * Z.height*sizeof(float), cudaMemcpyHostToDevice);    
    sigmoidForwardActivation(Zd,Ad);
    cudaMemcpy(A.elements, Ad.elements, Z.width * Z.height*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(Zd.elements);
    cudaFree(Ad.elements);

    return A;
}

__host__ Matrix generateWeights(int rows,int cols)
{
    //Matrix{w,h,*elements}
    Matrix W={cols,rows,allocateHostMemory(rows * cols)};

    float min=0.1f,max=1.9f;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.1, 1.9);
        
    for (int i = 0; i < rows*cols; i++)
    {            
        float f=0.0f;       
        while(min>abs(f) || abs(f)>max)
           f = distribution(generator);

        W.elements[i] = f;                        
    }

    return W;
}

__host__ float** generateWeights_(int rows,int cols)
{
    float **W;
    W=new float*[rows];

    float min=0.1f,max=1.9f;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.1, 1.9);

        
    W = new float *[rows];

    for (int i = 0; i < rows; i++)
    {
        W[i] = new float[cols];
        for (int j = 0; j < cols; j++)
        {         
            float f=0.0f;       
            while(min>abs(f) || abs(f)>max)
                f = distribution(generator);

            W[i][j] = f;                
        }
    }

    return W;
}

////////////////////////////////////////////////////////////////////////////
//testing
////////////////////////////////////////////////////////////////////////////
void testGenerateWeights(int rows,int cols)
{
    auto W=generateWeights(rows, cols);

    for (int i = 0; i < rows*cols; i++)
    {       
        std::cout << W.elements[i] << ((i + 1) % cols == 0 ? "\n" : ",");    
    }
    std::cout << "\n";
}
