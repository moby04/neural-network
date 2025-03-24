#ifndef LSTMLAYER_H
#define LSTMLAYER_H

#include "StatefulLayer.h"

/**
 * @brief Long Short-Term Memory (LSTM) Layer.
 * 
 * This layer is useful for sequence modeling tasks where long-term dependencies are important.
 * It maintains both a hidden state and a cell state to capture long-term dependencies.
 * 
 * More details: https://en.wikipedia.org/wiki/Long_short-term_memory
 */
class LSTMLayer : public StatefulLayer {
private:
    Matrix W_f, W_i, W_c, W_o; // Weights for forget, input, cell, output gates
    Matrix U_f, U_i, U_c, U_o; // Recurrent weights
    Matrix b_f, b_i, b_c, b_o; // Biases
    Matrix hiddenState;
    Matrix cellState; // Stores long-term memory

public:
    // Constructor
    LSTMLayer(size_t inputSize, size_t hiddenSize);

    // State Management
    void resetStates(); // Resets hidden and cell states

    // Forward and Backward Propagation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradOutput) override;

    // Getters
    inline Matrix getHiddenState() const {
        return hiddenState;
    }

    inline Matrix getCellState() const {
        return cellState;
    }
};

#endif // LSTMLAYER_H
