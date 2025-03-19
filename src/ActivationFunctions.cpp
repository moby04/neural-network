#include "../include/ActivationFunctions.h"
#include <cmath>

// -------------------- Sigmoid Activation --------------------
Matrix SigmoidActivation::apply(const Matrix& input) const {
    return input.applyFunction([](float x) { return 1.0f / (1.0f + exp(-x)); });
}

// -------------------- Swish Activation ----------------------
Matrix SwishActivation::apply(const Matrix& input) const {
    return input.applyFunction([](float x) { return x / (1.0f + exp(-x)); });
}

// -------------------- ReLU Activation -----------------------
Matrix ReLUActivation::apply(const Matrix& input) const {
    return input.applyFunction([](float x) { return x > 0 ? x : 0; });
}

// -------------------- Leaky ReLU Activation -----------------
Matrix LeakyReLUActivation::apply(const Matrix& input) const {
    return input.applyFunction([](float x) { return x > 0 ? x : 0.01f * x; });
}

// -------------------- Tanh Activation -----------------------
Matrix TanhActivation::apply(const Matrix& input) const {
    return input.applyFunction([](float x) { return tanh(x); });
}

// -------------------- Hard Tanh Activation ------------------
Matrix HardTanhActivation::apply(const Matrix& input) const {
    return input.applyFunction([](float x) { return (x < -1) ? -1 : (x > 1) ? 1 : x; });
}