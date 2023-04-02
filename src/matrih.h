#ifndef HMATRIX
#define HMATRIX

struct Matrix
{
    int width;
    int height;
    float *elements=nullptr;

    Matrix transpose();
    Matrix dot(Matrix B);
    Matrix multiply(Matrix B);
    
    float maxElement();
    int indexOfmax();

    Matrix operator+(const Matrix &B);
    Matrix operator+(const float scalar);    
    
    Matrix operator-(const Matrix &B);
    Matrix operator-(const float scalar);
    
    Matrix operator*(const float scalar);

    Matrix& operator=(const Matrix &B);

    Matrix operator[](const int index);

    void print();
};

Matrix operator+(const float scalar, const Matrix &mat);
Matrix operator-(const float scalar, const Matrix &mat);

typedef struct Matrix Matrix;

#endif
