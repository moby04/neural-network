#include "../include/ActivationFunctions.h"
#include <cmath>

// -------------------- Sigmoid Activation --------------------
Matrix SigmoidActivation::apply(const Matrix& input) const {
    return input.applyFunction([](float x) { return 1.0f / (1.0f + exp(-x)); });
}

Matrix SigmoidActivation::applyDerivative(const Matrix& input) const {
    Matrix sigmoidOut = SigmoidActivation().apply(input);
    Matrix ones(sigmoidOut.getRows(), sigmoidOut.getCols(), "Ones");
    ones.setData(1.0f);
    return sigmoidOut * (ones - sigmoidOut) ;
}


// -------------------- Swish Activation ----------------------
Matrix SwishActivation::apply(const Matrix& input) const {
    return input * SigmoidActivation::apply(input); 
}

Matrix SwishActivation::applyDerivative(const Matrix& input) const {
    SigmoidActivation sigmoid;
    return sigmoid.apply(input) + (input * sigmoid.applyDerivative(input)); 
}

// -------------------- ReLU Activation -----------------------
Matrix ReLUActivation::apply(const Matrix& input) const {
    return input.applyFunction([](float x) { return x > 0 ? x : 0; });
}

Matrix ReLUActivation::applyDerivative(const Matrix& input) const {
    return input.applyFunction([](float x) { return x > 0 ? 1 : 0; });
}

// -------------------- Leaky ReLU Activation -----------------
Matrix LeakyReLUActivation::apply(const Matrix& input) const {
    return input.applyFunction([](float x) { return x > 0 ? x : 0.01f * x; });
}

Matrix LeakyReLUActivation::applyDerivative(const Matrix& input) const {
    return input.applyFunction([](float x) { return x >= 0 ? 1 : 0.01; });
}

// -------------------- Tanh Activation -----------------------
Matrix TanhActivation::apply(const Matrix& input) const {
    return input.applyFunction([](float x) { return tanh(x); });
}

Matrix TanhActivation::applyDerivative(const Matrix& input) const {
    Matrix tanhOut = apply(input);
    Matrix ones(tanhOut.getRows(), tanhOut.getCols(), "Ones");
    ones.setData(1.0f);
    return ones - (tanhOut * tanhOut);
}

// -------------------- Hard Tanh Activation ------------------
Matrix HardTanhActivation::apply(const Matrix& input) const {
    return input.applyFunction([](float x) { return (x < -1) ? -1 : (x > 1) ? 1 : x; });
}

Matrix HardTanhActivation::applyDerivative(const Matrix& input) const {
    return input.applyFunction([](float x) { return (x > -1 && x < 1) ? 1 : 0; });
}