#include "feedforwardnn.h"


Matrix feedForward(Matrix x,Matrix w1,Matrix w2)
{
    //hidden layer
    auto z1 = x.dot(w1);     // input from layer 1
    auto a1 = sigmoid(z1);   // output of layer 2
     
    //Output layer
    auto z2 = a1.dot(w2);    // input of out layer
    auto a2 = sigmoid(z2);   // output of out layer
    return a2;
}