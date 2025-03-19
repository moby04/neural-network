#include "../include/Layer.h"

Layer::Layer(size_t neurons, std::shared_ptr<ActivationFunction> activationFunc)
    : weights(neurons, 0), biases(neurons, 1), activation(std::move(activationFunc)) {}

const Matrix& Layer::getWeights() const {
    return weights;
}

const Matrix& Layer::getBiases() const {
    return biases;
}

void Layer::setWeights(const Matrix& w) {
    if (w.getRows() != weights.getRows() || w.getCols() != weights.getCols()) {
        throw std::invalid_argument("Weight matrix dimensions do not match.");
    }
    weights = w;
}

void Layer::setBiases(const Matrix& b) {
    if (b.getRows() != biases.getRows() || b.getCols() != biases.getCols()) {
        throw std::invalid_argument("Bias matrix dimensions do not match.");
    }
    biases = b;
}
