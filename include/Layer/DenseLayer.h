#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "Layer.h"
#include <memory>

// Dense Layer (Fully Connected Layer)
class DenseLayer : public Layer {
private:
    Matrix inputCache;
public:
    DenseLayer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc);
    Matrix forward(const Matrix& input) override;
    Matrix backward(const Matrix& gradient) override;
};

#endif