#include <tuple>
#include "matrix.h"
#include "activationf.h"
#include "loss.h"

using namespace std;

tuple<Matrix, Matrix> backPropagation(Matrix X, Matrix Y, Matrix W1, Matrix W2, float alpha);
