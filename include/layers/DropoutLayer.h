#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "Layer.h"
#include <random>

/**
 * @brief Dropout Layer - Disables neurons during training to prevent overfitting.
 * 
 * This layer randomly disables neurons during training to prevent overfitting.
 * It is useful for regularizing neural networks and improving generalization.
 * 
 * More details: https://en.wikipedia.org/wiki/Dropout_(neural_networks)
 */
class DropoutLayer : public Layer {
private:
    float dropoutRate;
    std::mt19937 rng;
    Matrix dropoutMask;  // Stores dropped neurons (1 = active, 0 = dropped)

public:
    // Constructor
    DropoutLayer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc, float dropoutRate);

    // Forward and Backward Propagation
    Matrix forward(const Matrix& input) override;
    inline Matrix backward(const Matrix& gradient) override {
        return gradient * dropoutMask;  // Zero out gradients for dropped neurons
    }
};

#endif // DROPOUT_LAYER_H
