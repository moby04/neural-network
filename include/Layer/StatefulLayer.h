#ifndef STATEFUL_LAYER_H
#define STATEFUL_LAYER_H

#include "Layer.h"
#include "../Matrix.h"

/**
 * @brief Abstract base class for stateful layers.
 * 
 * Stateful layers maintain an internal state that is updated during forward passes and used during backward passes.
 * Examples include RNN, LSTM, and GRU layers.
 */
class StatefulLayer : public Layer {
protected:
    Matrix inputCache;

public:
    // Constructor and Destructor
    StatefulLayer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc)
        : Layer(inputSize, neurons, activationFunc), inputCache(inputSize, 1) {}

    virtual ~StatefulLayer() = default;

    // State Management
    /**
     * @brief Resets the internal state (must be implemented by derived classes).
     */
    virtual void resetStates() = 0;

    /**
     * @brief Clears inputCache (used in resetStates implementations).
     */
    void clearInputCache() {
        inputCache.setData(0.0);
    }

    // Getters
    /**
     * @brief Getter for input cache.
     * 
     * @return The input cache matrix.
     */
    const Matrix& getInputCache() const {
        return inputCache;
    }
};

#endif // STATEFUL_LAYER_H
