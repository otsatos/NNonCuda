#include<tuple>
#include<vector>
#include <numeric>
#include "matrix.h"
#include "feedforwardnn.h"
#include "backpropagation.h"
#include "loss.h"

tuple<vector<float>, vector<float>, Matrix, Matrix> train(Matrix X, Matrix Y, Matrix W1, Matrix W2, float alpha, int epoch);
