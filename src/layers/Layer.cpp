#include "../../include/layers/Layer.h"

// Getters
const Matrix& Layer::getWeights() const {
    return weights;
}

const Matrix& Layer::getBiases() const {
    return biases;
}

const Matrix& Layer::getInputCache() const {
    return inputCache;
}

// Setters
Layer& Layer::setWeights(const Matrix& w) {
    if (w.getRows() != weights.getRows() || w.getCols() != weights.getCols()) {
        throw std::invalid_argument("Weight matrix dimensions do not match.");
    }
    weights = w;
    return *this;
}

Layer& Layer::setBiases(const Matrix& b) {
    if (b.getRows() != biases.getRows() || b.getCols() != biases.getCols()) {
        throw std::invalid_argument("Bias matrix dimensions do not match.");
    }
    biases = b;
    return *this;
}