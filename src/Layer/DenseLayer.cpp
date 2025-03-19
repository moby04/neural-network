#include "../../include/Layer/DenseLayer.h"

// -------------------- DenseLayer Constructor --------------------
DenseLayer::DenseLayer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc)
    : Layer(inputSize, neurons, std::move(activationFunc)), inputCache(inputSize, 1)  {
    weights.randomize(-1.0f, 1.0f);
    biases.randomize(-1.0f, 1.0f);
}

// -------------------- Forward Propagation --------------------
Matrix DenseLayer::forward(const Matrix& input) {
    inputCache = input;
    Matrix weightedSum = weights.multiply(input, false) + biases;  // W * X + B
    return activation->apply(weightedSum);  // Apply activation function
}

// -------------------- Backward Propagation --------------------
Matrix DenseLayer::backward(const Matrix& gradient) {
    // Compute activation gradient
    Matrix activationGradient = activation->applyDerivative(forward(inputCache));  // ✅ Now works!

    // Compute weight and bias gradients
    Matrix weightGradient = gradient.multiply(inputCache.transpose(), false);
    Matrix biasGradient = gradient;

    // Update weights and biases (gradient descent)
    weights = weights - (weightGradient * 0.01);
    biases = biases - (biasGradient * 0.01);

    // Compute gradient for previous layer
    return weights.transpose().multiply(gradient, false);
}