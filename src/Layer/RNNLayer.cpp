#include "../../include/Layer/RNNLayer.h"
#include <cmath>

// Constructor
RNNLayer::RNNLayer(size_t inputSize, size_t hiddenSize)
        : StatefulLayer(inputSize, hiddenSize, nullptr),
        W_x(inputSize, hiddenSize, "W_x"),
        W_h(hiddenSize, hiddenSize, "W_h"),
        b(1, hiddenSize, "b"),
        hiddenState(1, hiddenSize, "hiddenState") {
    W_x.randomize();
    W_h.randomize();
    b.randomize();
    hiddenState.setData(0.0);
}

// State Management
void RNNLayer::resetStates() {
    hiddenState.setData(0.0);
    clearInputCache();
}

// Forward Propagation
Matrix RNNLayer::forward(const Matrix& input) {
    inputCache = input; 

    // Compute new hidden state
    hiddenState = (input.multiply(W_x, false) + hiddenState.multiply(W_h, false) + b).applyFunction([](float x) {
        return tanh(x);
    });

    return hiddenState; // Output is also the hidden state
}

// Backward Propagation
Matrix RNNLayer::backward(const Matrix& gradOutput) {
    Matrix ones(hiddenState.getRows(), hiddenState.getCols(), "Ones");
    ones.setData(1.0); 

    Matrix squaredHidden = hiddenState * hiddenState;
    Matrix dHidden = gradOutput * (ones - squaredHidden);

    return dHidden.multiply(W_x.transpose(), false);
}

// Getters
Matrix RNNLayer::getHiddenState() const {
    return hiddenState;
}
