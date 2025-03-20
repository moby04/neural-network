#ifndef LSTMLAYER_H
#define LSTMLAYER_H

#include "StatefulLayer.h"

class LSTMLayer : public StatefulLayer {
private:
    Matrix W_f, W_i, W_c, W_o; // Weights for forget, input, cell, output gates
    Matrix U_f, U_i, U_c, U_o; // Recurrent weights
    Matrix b_f, b_i, b_c, b_o; // Biases
    Matrix hiddenState;
    Matrix cellState; // Stores long-term memory

public:
    LSTMLayer(size_t inputSize, size_t hiddenSize);

    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradOutput) override;
    void resetStates(); // Resets hidden and cell states

    Matrix getHiddenState() const;
    Matrix getCellState() const;
};

#endif // LSTMLAYER_H
