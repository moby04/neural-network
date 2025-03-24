#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "../Matrix.h"
#include "StatefulLayer.h"
#include <memory>

// Dense Layer (Fully Connected Layer)
/**
 * @brief Dense (Fully Connected) Layer.
 * 
 * This layer is a basic building block of neural networks where each neuron is connected to every neuron in the previous layer.
 * It is useful for tasks where a fully connected network is needed, such as classification and regression.
 * 
 * More details: https://en.wikipedia.org/wiki/Feedforward_neural_network
 */
class DenseLayer : public StatefulLayer {
private:
    Matrix weights, biases;
public:
    // Constructor
    DenseLayer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc);

    // State Management
    void resetStates() override;

    // Forward and Backward Propagation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
};

#endif