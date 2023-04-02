#include "backpropagation.h"


tuple<Matrix,Matrix> backPropagation(Matrix X,Matrix Y, Matrix W1, Matrix W2, float alpha)
{
    // hidden layer
    auto Z1 = X.dot(W1); 
    auto A1 = sigmoid(Z1);
     
    //Output layer
    auto Z2 = A1.dot(W2);
    auto A2 = sigmoid(Z2);

    // errors in output layer
    auto D2 =(A2-Y);

    auto D1 = ((W2.dot(D2.transpose())).transpose()).multiply(A1.multiply(1-A1));
  
    //Gradients for W1 and W2
    auto W1adj = X.transpose().dot(D1);
    auto W2adj = A1.transpose().dot(D2);
     
    //Updating parameters
    auto W1new = W1-(W1adj*alpha);
    auto W2new = W2-(W2adj*alpha);

    return {W1new,W2new};
}