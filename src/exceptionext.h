#ifndef  HEXCEPTIONS
#define HEXCEPTIONS

#include <exception>
#include <iostream>
#include <cuda_runtime.h>

using namespace std;

class ExceptionExt : std::exception
{
private:
    const char *ex;

public:
    ExceptionExt(const char *ex) : ex(ex)
    {}

    virtual const char *what() const throw()
    {
        return ex;
    }

    static void throwCudaError(const char *ex)
    {
        cudaError_t derror = cudaGetLastError();
        if (derror != cudaSuccess)
        {
            std::cerr << derror << ": " << ex;
            throw ExceptionExt(ex);
        }
    }
};

#endif