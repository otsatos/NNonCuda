#include "matrixoperations.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix X Matrix Multiplication(dot product) C=αop(A)op(B)+βC 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kdotProduct(Matrix A, Matrix B, Matrix C)
{
    float val = 0;    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int e = 0; e < A.width; ++e)
      val += A.elements[row * A.width + e] * B.elements[e * B.width + col];

    C.elements[row * C.width + col] = val; 
}

__host__ void dotProduct(Matrix A,Matrix B,Matrix C)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((B.width/dimBlock.x>1?B.width/dimBlock.x:1), (A.height/dimBlock.y>1?A.height/dimBlock.y:1));
    kdotProduct<<<dimGrid, dimBlock>>>(A, B, C);   
}

void dotProductCublas(Matrix A,Matrix B,Matrix C)
{      
    // https://docs.nvidia.com/cuda/cublas/   
    // CUBLAS_OP_N : the non-transpose operation is selected
    // m :number of rows of matrix op(A) and C
    // n :number of columns of matrix op(B) and C
    // k : number of columns of op(A) and rows of op(B)
    
    // cublass considers matrices stored in column-major format, CUBLAS_OP_T transposes the input to row-major matrix 
    // Instead of transposition,we change the orders of arguments, first Matrix B instead of A etc...
    //https://peterwittek.com/cublas-matrix-c-style.html
    
    int m = B.width, n = A.height, k = A.width;
    
    const float alpha = 1.0f, beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);
         
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, B.elements, m, A.elements, B.height, &beta, C.elements, m); 

    cublasDestroy(handle); 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix Trasposition 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kmatrixTranspose(Matrix A,Matrix At)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  //input row,col
  int row=i / At.width;
  int col=i % At.width;
  
  //transposed index
  int j=col * At.height + row;

  if (i < A.width * A.height && j< At.width * At.height) 
    At.elements[j] = A.elements[i];
}

__host__ void matrixTranspose(Matrix A,Matrix At)
{     
  dim3 dimBlock(16, 16);
  dim3 dimGrid((At.width/dimBlock.x>1?At.width/dimBlock.x:1), (A.height/dimBlock.y>1?A.height/dimBlock.y:1));

  kmatrixTranspose<<<dimGrid, dimBlock>>>(A,At);  
}

void matrixTransposeCublas(Matrix A,Matrix At)
{
     // https://docs.nvidia.com/cuda/cublas/   
    // CUBLAS_OP_N : the non-transpose operation is selected
    // m :number of rows of matrix op(A) and C
    // n :number of columns of matrix op(B) and C
    // k : number of columns of op(A) and rows of op(B)

    int m = At.height, n = At.width;
    const float alpha = 1.0f, beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);
         
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, A.elements, n, &beta, NULL, m, At.elements, m);

    cublasDestroy(handle); 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Element-wise Matrix Multiplication  
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void kmatrixMultiplyElWise(Matrix A, Matrix B,Matrix C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < A.width * A.height) 
    {
	    C.elements[i] = A.elements[i]*B.elements[i];
    }
}

__host__ void matrixMultiplyElWise(Matrix A, Matrix B,Matrix C)
{
    dim3 blockSize(256);
    dim3 numOfBlocks((A.width * A.height + blockSize.x - 1) / blockSize.x);

    kmatrixMultiplyElWise<<<numOfBlocks, blockSize>>>(A, B,C);	   
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Test above functions
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void testmatrixMultiplyElWise()
{
    int n=8;
    Matrix A={2,4,allocateHostMemory(n)};
    Matrix B={2,4,allocateHostMemory(n)};
    Matrix C={2,4,allocateHostMemory(n)};

    float fv=2.0f;
    for (int i=0;i<n;i++) {A.elements[i]=fv;fv+=2.0f;}

    fv=1.0f;    
    for (int i=0;i<n;i++) {B.elements[i]=fv;fv+=2.0f;}
        
    Matrix Ad={2,4,allocateDeviceMemory(n)};
    Matrix Bd={2,4,allocateDeviceMemory(n)};    
    Matrix Cd={2,4,allocateDeviceMemory(n)};    

	  cudaMemcpy(Ad.elements, A.elements, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bd.elements, B.elements, n*sizeof(float), cudaMemcpyHostToDevice);
        
    matrixMultiplyElWise(Ad,Bd,Cd);

    cudaMemcpy(C.elements,Cd.elements,n*sizeof(float), cudaMemcpyDeviceToHost);
   
    C.print(); 
}

void testMAtrixTranspose()
{
    int n=12;
    Matrix A ={4,3,allocateHostMemory(n)};    
    Matrix At={3,4,allocateHostMemory(n)};    

    Matrix Ad  ={4,3,allocateDeviceMemory(n)};
    Matrix Atd ={3,4,allocateDeviceMemory(n)};    
    
    for (int i=0;i<n;i++){A.elements[i]=i+1;}
    for (int i=0;i<n;i++){At.elements[i]=0.0f;}
    for (int i=0;i<n;i++){std::cout << A.elements[i] << ((i+1)%3!=0?",":"\n");}
    
    cudaMemcpy(Ad.elements,A.elements,n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Atd.elements,At.elements,n*sizeof(float), cudaMemcpyHostToDevice);//unnecessary
    
    //matrixTranspose(Ad,Atd);
    matrixTransposeCublas(Ad,Atd);
    cudaMemcpy(At.elements,Atd.elements,n*sizeof(float), cudaMemcpyDeviceToHost);
    At.print();

    cudaFree(Ad.elements);
    cudaFree(Atd.elements);    

    if (A.elements) free(A.elements);
    if (At.elements) free(At.elements);    
}

void testDotProduct()
{
  int n=16;
  Matrix A={4,4,allocateHostMemory(n)};    
  Matrix B={4,4,allocateHostMemory(n)};  
  Matrix C={4,4,allocateHostMemory(n)};  
  
  Matrix Ad={4,4,allocateDeviceMemory(n)};
  Matrix Bd={4,4,allocateDeviceMemory(n)};
  Matrix Cd={4,4,allocateDeviceMemory(n)};
  
  float fv=2.0f;
  for (int i=0;i<n;i++) {A.elements[i]=fv;fv+=2.0f;}
  fv=1.0f;    
  for (int i=0;i<n;i++) {B.elements[i]=fv;fv+=2.0f;}
  for (int i=0;i<n;i++) {C.elements[i]=0.0f;}       

  cudaMemcpy(Ad.elements, A.elements, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Bd.elements, B.elements, n*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Cd.elements, C.elements, n*sizeof(float), cudaMemcpyHostToDevice);
 
  //dotProduct(Ad,Bd,Cd); 
  dotProductCublas(Ad,Bd,Cd);
  cudaMemcpy(C.elements,Cd.elements,n*sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "A" << "\n";   
  A.print();

  std::cout << "B" << "\n";   
  B.print();

  std::cout << "C" << "\n";  
  C.print();
  
  cudaFree(Ad.elements);
  cudaFree(Bd.elements);
  cudaFree(Cd.elements);   

  if (A.elements) free(A.elements);
  if (B.elements) free(B.elements);
  if (C.elements) free(C.elements);
}