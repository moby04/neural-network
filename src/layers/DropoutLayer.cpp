#include "../../include/layers/DropoutLayer.h"

// Constructor
DropoutLayer::DropoutLayer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc, float dropoutRate)
    : Layer(inputSize, neurons, std::move(activationFunc)), dropoutRate(dropoutRate), rng(std::random_device{}()), dropoutMask(inputSize, 1)  {
    weights.randomize(-1.0f, 1.0f);
    biases.randomize(-1.0f, 1.0f);
}

// Forward Propagation
Matrix DropoutLayer::forward(const Matrix& input) {
    Matrix output = input;
    dropoutMask = Matrix(input.getRows(), input.getCols(), "DropoutMask");  // Store active neurons (1 = active, 0 = dropped)
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (size_t i = 0; i < output.getRows(); ++i) {
        for (size_t j = 0; j < output.getCols(); ++j) {
            if (dist(rng) < dropoutRate) {
                dropoutMask(i, j) = 0.0;  // Mark as dropped
                output(i, j) = 0.0;  // Disable neuron
            } else {
                dropoutMask(i, j) = 1.0;  // Mark as active
                output(i, j) /= (1.0 - dropoutRate);  // Scale remaining neurons
            }
        }
    }

    return output;
}
