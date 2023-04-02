#include "matrix.h"
#include "matrixoperations.h"
#include "memallocations.h"
#include <limits>

using namespace std;

Matrix Matrix::transpose()
{
    int n=width*height;
    Matrix At ={height,width,allocateHostMemory(n)};

    Matrix Ad ={width, height, allocateDeviceMemory(n)};
    Matrix Atd={height, width, allocateDeviceMemory(n)};

    cudaMemcpy(Ad.elements, elements, n * sizeof(float), cudaMemcpyHostToDevice);
    matrixTransposeCublas(Ad,Atd);
    cudaMemcpy(At.elements,Atd.elements,n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(Ad.elements);
    cudaFree(Atd.elements);  

    return At;
}

Matrix Matrix::dot(Matrix B)
{
    //A:this 
    //C(Bw,Ah)=A(w,h).B(w,h)
    
    if (width!=B.height)  throw std::invalid_argument( "Number of columns in 1st Matrix must be equal to the number of rows in the 2nd Matrix!" );

    int Cn=height*B.width;    

    Matrix Ad={width,height,allocateDeviceMemory(width*height)};
    Matrix Bd={B.width,B.height,allocateDeviceMemory(B.width*B.height)};

    Matrix C={B.width,height,allocateHostMemory(Cn)};  
    Matrix Cd={B.width,height,allocateDeviceMemory(Cn)};  

    cudaMemcpy(Ad.elements, elements, width*height*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bd.elements, B.elements, B.width*B.height*sizeof(float), cudaMemcpyHostToDevice);
    
    dotProductCublas(Ad,Bd,Cd);
    //dotProduct(Ad,Bd,Cd);
    cudaMemcpy(C.elements,Cd.elements,Cn*sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(Ad.elements);
    cudaFree(Bd.elements);  
    cudaFree(Cd.elements);  

    return C;
}

Matrix Matrix::multiply(Matrix B)
{
    int n=width*height;

    Matrix C={width,height,allocateHostMemory(n)};
    
    Matrix Ad={width,height,allocateDeviceMemory(n)};
    Matrix Bd={width,height,allocateDeviceMemory(B.width * B.height)};
    Matrix Cd={width,height,allocateDeviceMemory(n)};

    cudaMemcpy(Ad.elements, elements, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bd.elements, B.elements, B.width*B.height * sizeof(float), cudaMemcpyHostToDevice);

    matrixMultiplyElWise(Ad,Bd,Cd);
    cudaMemcpy(C.elements,Cd.elements,n * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(Ad.elements);
    cudaFree(Bd.elements);  
    cudaFree(Cd.elements);  

    return C;    
}

float  Matrix::maxElement()
{
    float m=std::numeric_limits<float>::min();
    for(int i=0;i<width*height;i++)
    {
        if (elements[i]>m) 
            m=elements[i];
    }
        
    return m;    
}

int  Matrix::indexOfmax()
{
    float m=std::numeric_limits<float>::min();    
    int idx=-1;
     for(int i=0;i<width*height;i++)
     {
        if (elements[i]>m) 
        {
            m=elements[i];
            idx=i;
        }
     }

    return idx;    
}

////////////////////////////////////////////////////////////////////////
//operators overloading
////////////////////////////////////////////////////////////////////////
Matrix Matrix::operator+(const Matrix &B)
{
    Matrix C = {width,height,allocateHostMemory(width*height)};
    for(int i=0;i<(width*height);++i) C.elements[i]=elements[i] + B.elements[i];
    return C; 
}
Matrix Matrix::operator+(const float scalar)
{
    Matrix C = {width,height,allocateHostMemory(width*height)};
    for(int i=0;i<(width*height);++i) C.elements[i]=elements[i] + scalar;
    return C; 
}
Matrix operator+(const float scalar,const Matrix &A)
{
    Matrix C = {A.width,A.height,allocateHostMemory(A.width*A.height)};
    for(int i=0;i<(A.width*A.height);++i) C.elements[i]=scalar + A.elements[i];
    return C; 
}

Matrix Matrix::operator-(const Matrix &B)
{
    Matrix C = {width,height,allocateHostMemory(width*height)};
    for(int i=0;i<(width*height);++i) C.elements[i]=elements[i] - B.elements[i];
    return C; 
}
Matrix Matrix::operator-(const float scalar)
{
    Matrix C = {width,height,allocateHostMemory(width*height)};
    for(int i=0;i<(width*height);++i) C.elements[i]=elements[i] - scalar;
    return C; 
}
Matrix operator-(const float scalar,const Matrix &A)
{
    Matrix C = {A.width,A.height,allocateHostMemory(A.width*A.height)};
    for(int i=0;i<(A.width*A.height);++i) C.elements[i]=scalar - A.elements[i];
    return C; 
}
Matrix Matrix::operator*(const float scalar)
{
    Matrix C = {width,height,allocateHostMemory(width*height)};
    for(int i=0;i<(width*height);++i) C.elements[i]=elements[i] * scalar;
    return C; 
}

Matrix& Matrix::operator=(const Matrix &B)
{
    if (width!=0 && &elements!=nullptr) free(elements);

    width = B.width;
    height=B.height;
    
    elements = B.elements;
    return *this;
}

Matrix Matrix::operator[](const int index)
{
    Matrix S={width,1,allocateHostMemory(width)};
    int i=0;
    for(int j=(index * width);j< ((index+1)*width);++j)
    {
      S.elements[i]=elements[j];
      i++;
    }
    return S;
}

void Matrix::print()
    {
        std::cout << "Matrix(" << height << " rows," << width <<" columns)"<< "\n";       
         
        for (int i=0;i<(width*height);i++)
        {                                             
            std::cout << elements[i];
            std::cout << ((i+1)%width==0?"\n":",");
        }
    }
