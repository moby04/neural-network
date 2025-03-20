#ifndef GRU_LAYER_H
#define GRU_LAYER_H

#include "../Matrix.h"
#include "../ActivationFunctions/ActivationFunctions.h"
#include "StatefulLayer.h"

class GRULayer : public StatefulLayer {
private:
    Matrix W_z, W_r, W_h; // Weights for update, reset, and candidate activation
    Matrix U_z, U_r, U_h; // Recurrent weights
    Matrix b_z, b_r, b_h; // Biases
    Matrix hiddenState;    // Hidden state

public:
    GRULayer(size_t inputSize, size_t hiddenSize);

    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradOutput) override;
    
    void resetStates() override;

    void resetHiddenState();
    Matrix getHiddenState() const;
};

#endif // GRU_LAYER_H
