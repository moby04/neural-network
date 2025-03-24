#ifndef RNN_LAYER_H
#define RNN_LAYER_H

#include "StatefulLayer.h"
#include "../Matrix.h"

/**
 * @brief Recurrent Neural Network (RNN) Layer.
 * 
 * This layer is useful for sequence modeling tasks such as time series prediction and natural language processing.
 * It maintains a hidden state that is updated at each time step.
 * 
 * More details: https://en.wikipedia.org/wiki/Recurrent_neural_network
 */

/**
 * A Simple RNN unit operates using the following update rule:
 * ht = tanh(Wx * xt + Wh * h(t-1) + b)
 * 
 * where:
 * ht - is the hidden state at time t
 * xt - is the input at time t
 * Wx - is the input-to-hidden weight matrix
 * Wh - is the hidden-to-hidden weight matrix
 * b - is the bias
 * h(t-1) - is the hidden state at time t-1
 */
class RNNLayer : public StatefulLayer {
    private:
        Matrix W_x, W_h, b, hiddenState;
    
    public:
        // Constructor
        RNNLayer(size_t inputSize, size_t hiddenSize);
    
        // State Management
        void resetStates() override {
            hiddenState.setData(0.0);
            clearInputCache();  
        }
    
        // Forward and Backward Propagation
        Matrix forward(const Matrix& input) override;
        Matrix backward(const Matrix& gradOutput) override;

        // Getters
        inline Matrix getHiddenState() const {
            return hiddenState; 
        }
};

#endif