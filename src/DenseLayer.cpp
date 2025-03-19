#include "../include/DenseLayer.h"

// -------------------- DenseLayer Constructor --------------------
DenseLayer::DenseLayer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc)
    : Layer(neurons, std::move(activationFunc)) {
    weights = Matrix(neurons, inputSize);
    biases = Matrix(neurons, 1);
    weights.randomize(-1.0f, 1.0f);
    biases.randomize(-1.0f, 1.0f);
}

// -------------------- Forward Propagation --------------------
Matrix DenseLayer::forward(const Matrix& input) {
    Matrix weightedSum = weights.multiply(input, false) + biases;  // W * X + B
    return activation->apply(weightedSum);  // Apply activation function
}