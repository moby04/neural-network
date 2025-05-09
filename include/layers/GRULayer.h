#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "../matrix/Matrix.h"
#include "../activations/ActivationFunctions.h"
#include "StatefulLayer.h"

/**
 * @brief Gated Recurrent Unit (GRU) Layer.
 * 
 * This layer is useful for sequence modeling tasks and is a simpler alternative to LSTM.
 * It maintains a hidden state and uses update and reset gates to control the flow of information.
 * 
 * More details: https://en.wikipedia.org/wiki/Gated_recurrent_unit
 */
class GRULayer : public StatefulLayer {
private:
    Matrix W_z, W_r, W_h; // Weights for update, reset, and candidate activation
    Matrix U_z, U_r, U_h; // Recurrent weights
    Matrix b_z, b_r, b_h; // Biases
    Matrix hiddenState;    // Hidden state

public:
    // Constructor
    GRULayer(size_t inputSize, size_t hiddenSize);

    // State Management
    inline GRULayer& resetStates() override {
        resetHiddenState();
        clearInputCache();
        return *this;
    }

    GRULayer& resetHiddenState() {
        hiddenState.setData(0.0);
        return *this;
    }

    // Forward and Backward Propagation
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradOutput) override;

    // Getters
    inline Matrix getHiddenState() const {
        return hiddenState;
    }
};

#endif // GRU_LAYER_H
