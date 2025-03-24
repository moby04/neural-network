#include "../../include/layers/GRULayer.h"
#include <cmath>

// Constructor
GRULayer::GRULayer(size_t inputSize, size_t hiddenSize)
        : StatefulLayer(inputSize, hiddenSize, nullptr),
        W_z(inputSize, hiddenSize, "W_z"), W_r(inputSize, hiddenSize, "W_r"), W_h(inputSize, hiddenSize, "W_h"),
        U_z(hiddenSize, hiddenSize, "U_z"), U_r(hiddenSize, hiddenSize, "U_r"), U_h(hiddenSize, hiddenSize, "U_h"),
        b_z(1, hiddenSize, "b_z"), b_r(1, hiddenSize, "b_r"), b_h(1, hiddenSize, "b_h"),
        hiddenState(1, hiddenSize, "hiddenState") {
    W_z.randomize(); W_r.randomize(); W_h.randomize();
    U_z.randomize(); U_r.randomize(); U_h.randomize();
    b_z.randomize(); b_r.randomize(); b_h.randomize();
    hiddenState.setData(0.0);
}

// Forward Propagation
Matrix GRULayer::forward(const Matrix& input) {
    if (input.isEmpty()) {
        throw std::runtime_error("Forward pass: Input matrix is empty.");
    }
    inputCache = input;
    SigmoidActivation sigmoid;
    TanhActivation tanh;

    Matrix z_t = sigmoid.apply(input.multiply(W_z, false) + hiddenState.multiply(U_z, false) + b_z);
    Matrix r_t = sigmoid.apply(input.multiply(W_r, false) + hiddenState.multiply(U_r, false) + b_r);

    Matrix h_tilde = tanh.apply(input.multiply(W_h, false) + (hiddenState * r_t).multiply(U_h, false) + b_h);

    Matrix ones(1, hiddenState.getCols(), "Ones");
    ones.setData(1.0);

    hiddenState = ((ones - z_t) * hiddenState) + (z_t * h_tilde);
    return hiddenState;
}

// Backward Propagation
Matrix GRULayer::backward(const Matrix& gradOutput) {
    if (inputCache.isEmpty(true)) {
        throw std::runtime_error("Backward pass: forward() must be called before backward().");
    }

    SigmoidActivation sigmoid;
    TanhActivation tanh;

    Matrix z_t = sigmoid.apply(inputCache.multiply(W_z, false) + hiddenState.multiply(U_z, false) + b_z);
    Matrix r_t = sigmoid.apply(inputCache.multiply(W_r, false) + hiddenState.multiply(U_r, false) + b_r);
    Matrix h_tilde = tanh.apply(inputCache.multiply(W_h, false) + (hiddenState * r_t).multiply(U_h, false) + b_h);

    // Compute gradients
    Matrix dH = gradOutput * (Matrix(1, hiddenState.getCols(), "Ones").setData(1.0) - z_t);
    Matrix dZ = gradOutput * (h_tilde - hiddenState);
    Matrix dR = dH * (hiddenState.multiply(U_h, false));

    // Weight updates
    W_z = W_z - (inputCache.transpose().multiply(dZ, false) * 0.01);
    W_r = W_r - (inputCache.transpose().multiply(dR, false) * 0.01);
    W_h = W_h - (inputCache.transpose().multiply(dH, false) * 0.01);

    return dH.multiply(W_z.transpose(), false);
}
