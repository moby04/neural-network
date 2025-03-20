#include "../../include/Layer/LSTMLayer.h"
#include "../../include/ActivationFunctions/ActivationFunctions.h"
#include <cmath>

LSTMLayer::LSTMLayer(size_t inputSize, size_t hiddenSize)
        : StatefulLayer(inputSize, hiddenSize, nullptr),
        W_f(inputSize, hiddenSize, "W_f"), W_i(inputSize, hiddenSize, "W_i"),
        W_c(inputSize, hiddenSize, "W_c"), W_o(inputSize, hiddenSize, "W_o"),
        U_f(hiddenSize, hiddenSize, "U_f"), U_i(hiddenSize, hiddenSize, "U_i"),
        U_c(hiddenSize, hiddenSize, "U_c"), U_o(hiddenSize, hiddenSize, "U_o"),
        b_f(1, hiddenSize, "b_f"), b_i(1, hiddenSize, "b_i"),
        b_c(1, hiddenSize, "b_c"), b_o(1, hiddenSize, "b_o"),
        hiddenState(1, hiddenSize, "hiddenState"),
        cellState(1, hiddenSize, "cellState") {
    W_f.randomize(); W_i.randomize(); W_c.randomize(); W_o.randomize();
    U_f.randomize(); U_i.randomize(); U_c.randomize(); U_o.randomize();
    b_f.randomize(); b_i.randomize(); b_c.randomize(); b_o.randomize();
    hiddenState.setData(0.0);
    cellState.setData(0.0);
}

Matrix LSTMLayer::forward(const Matrix& input) {
    if (input.isEmpty()) {
        throw std::runtime_error("Forward pass: Input matrix is empty.");
    }
    inputCache = input;

    SigmoidActivation sigmoid;
    TanhActivation tanh;

    // Forget Gate
    Matrix f_t = sigmoid.apply(input.multiply(W_f, false) + hiddenState.multiply(U_f, false) + b_f);
    
    // Input Gate
    Matrix i_t = sigmoid.apply(input.multiply(W_i, false) + hiddenState.multiply(U_i, false) + b_i);
    
    // Candidate Cell State
    Matrix c_tilde = tanh.apply(input.multiply(W_c, false) + hiddenState.multiply(U_c, false) + b_c);
    
    // Cell State
    cellState = (f_t * cellState) + (i_t * c_tilde);
    
    // Output Gate
    Matrix o_t = sigmoid.apply(input.multiply(W_o, false) + hiddenState.multiply(U_o, false) + b_o);
    
    // Hidden State
    hiddenState = o_t * tanh.apply(cellState);

    return hiddenState;
}

Matrix LSTMLayer::backward(const Matrix& gradOutput) {
    if (inputCache.isEmpty(true)) {
        throw std::runtime_error("Backward pass: forward() must be called before backward().");
    }
    if (hiddenState.isEmpty(true) || cellState.isEmpty(true)) {
        throw std::runtime_error("Backward pass: hiddenState and cellState must be initialized.");
    }
    if (gradOutput.isEmpty(true)) {
        throw std::runtime_error("Backward pass: gradOutput cannot be empty.");
    }
    if (gradOutput.getCols() != W_f.transpose().getRows()) {
        throw std::runtime_error("Backward pass: gradOutput dimensions do not match transposed weight dimensions.");
    }

    SigmoidActivation sigmoid;
    TanhActivation tanh;

    Matrix dO = gradOutput * tanh.apply(cellState);
    Matrix dC = gradOutput * hiddenState;
    Matrix dF = gradOutput * cellState;
    Matrix dI = gradOutput * dC;

    try {
        W_o = W_o - (inputCache.transpose().multiply(dO, false) * 0.01);
        W_f = W_f - (inputCache.transpose().multiply(dF, false) * 0.01);
        W_i = W_i - (inputCache.transpose().multiply(dI, false) * 0.01);
        W_c = W_c - (inputCache.transpose().multiply(dC, false) * 0.01);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Backward pass error: ") + e.what());
    }

    return gradOutput.multiply(W_f.transpose(), false);
}

void LSTMLayer::resetStates() {
    hiddenState.setData(0.0);
        cellState.setData(0.0);
        clearInputCache();
    }

Matrix LSTMLayer::getHiddenState() const {
    return hiddenState;
}

Matrix LSTMLayer::getCellState() const {
    return cellState;
}
