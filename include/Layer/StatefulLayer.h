#ifndef STATEFUL_LAYER_H
#define STATEFUL_LAYER_H

#include "Layer.h"
#include "../Matrix.h"

class StatefulLayer : public Layer {
protected:
    Matrix inputCache;

public:
    StatefulLayer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc)
        : Layer(inputSize, neurons, activationFunc), inputCache(inputSize, 1) {}

    virtual ~StatefulLayer() = default;

    // Resets the internal state (must be implemented by derived classes)
    virtual void resetStates() = 0;

    // Getter for input cache
    const Matrix& getInputCache() const {
        return inputCache;
    }

    // Clears inputCache (used in resetStates implementations)
    void clearInputCache() {
        inputCache.setData(0.0);
    }
};

#endif // STATEFUL_LAYER_H
