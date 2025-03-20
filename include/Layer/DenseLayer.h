#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "../Matrix.h"
#include "StatefulLayer.h"
#include <memory>

// Dense Layer (Fully Connected Layer)
class DenseLayer : public StatefulLayer {
private:
    Matrix weights, biases;
public:
    DenseLayer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;

    void resetStates() override;
};

#endif