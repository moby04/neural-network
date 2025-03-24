#include "../../include/activations/ActivationFunctions.h"
#include <cmath>

// -------------------- Sigmoid Activation --------------------
// Forward Propagation
Matrix SigmoidActivation::apply(const Matrix& input) const {
    return input.applyFunction([](double x) { return 1.0 / (1.0 + exp(-x)); });
}

// Backward Propagation
Matrix SigmoidActivation::applyDerivative(const Matrix& input) const {
    Matrix sigmoidOut = SigmoidActivation().apply(input);
    Matrix ones(sigmoidOut.getRows(), sigmoidOut.getCols(), "Ones");
    ones.setData(1.0);
    return sigmoidOut * (ones - sigmoidOut) ;
}

// -------------------- Swish Activation ----------------------
// Forward Propagation
Matrix SwishActivation::apply(const Matrix& input) const {
    return input * SigmoidActivation::apply(input); 
}

// Backward Propagation
Matrix SwishActivation::applyDerivative(const Matrix& input) const {
    SigmoidActivation sigmoid;
    return sigmoid.apply(input) + (input * sigmoid.applyDerivative(input)); 
}

// -------------------- ReLU Activation -----------------------
// Forward Propagation
Matrix ReLUActivation::apply(const Matrix& input) const {
    return input.applyFunction([](double x) { return x > 0 ? x : 0; });
}

// Backward Propagation
Matrix ReLUActivation::applyDerivative(const Matrix& input) const {
    return input.applyFunction([](double x) { return x > 0 ? 1 : 0; });
}

// -------------------- Leaky ReLU Activation -----------------
// Forward Propagation
Matrix LeakyReLUActivation::apply(const Matrix& input) const {
    return input.applyFunction([](double x) { return x > 0 ? x : 0.01 * x; });
}

// Backward Propagation
Matrix LeakyReLUActivation::applyDerivative(const Matrix& input) const {
    return input.applyFunction([](double x) { return x >= 0 ? 1 : 0.01; });
}

// -------------------- Tanh Activation -----------------------
// Forward Propagation
Matrix TanhActivation::apply(const Matrix& input) const {
    return input.applyFunction([](double x) { return tanh(x); });
}

// Backward Propagation
Matrix TanhActivation::applyDerivative(const Matrix& input) const {
    Matrix tanhOut = apply(input);
    Matrix ones(tanhOut.getRows(), tanhOut.getCols(), "Ones");
    ones.setData(1.0);
    return ones - (tanhOut * tanhOut);
}

// -------------------- Hard Tanh Activation ------------------
// Forward Propagation
Matrix HardTanhActivation::apply(const Matrix& input) const {
    return input.applyFunction([](double x) { return (x < -1) ? -1 : (x > 1) ? 1 : x; });
}

// Backward Propagation
Matrix HardTanhActivation::applyDerivative(const Matrix& input) const {
    return input.applyFunction([](double x) { return (x > -1 && x < 1) ? 1 : 0; });
}