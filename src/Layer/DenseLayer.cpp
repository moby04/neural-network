#include "../../include/Layer/DenseLayer.h"

// Constructor
DenseLayer::DenseLayer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc)
        : StatefulLayer(inputSize, neurons, activationFunc),
          weights(neurons, inputSize, "weights"),
          biases(neurons, 1, "biases") {
    weights.randomize();
    biases.randomize();
}

// Forward Propagation
Matrix DenseLayer::forward(const Matrix& input) {
    inputCache = input;

    Matrix output = (weights.multiply(input, false) + biases);

    return activation->apply(output);
}

// Backward Propagation
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