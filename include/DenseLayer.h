#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "Layer.h"
#include "Matrix.h"
#include <memory>

// Dense Layer (Fully Connected Layer)
class DenseLayer : public Layer {
public:
    DenseLayer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc);
    Matrix forward(const Matrix& input) override;
};

#endif