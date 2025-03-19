#ifndef LAYER_H
#define LAYER_H

#include "../ActivationFunctions/ActivationFunctions.h"
#include "../Matrix.h"
#include <functional>
#include <memory>

class Layer {
protected:
    Matrix weights;
    Matrix biases;
    std::shared_ptr<ActivationFunction> activation;

public:
    Layer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc)
        : weights(neurons, inputSize), biases(neurons, 1), activation(activationFunc) {}
    virtual ~Layer() = default;

    virtual Matrix forward(const Matrix& input) = 0;
    virtual Matrix backward(const Matrix& gradient) = 0; 

    // Getters for weights and biases
    const Matrix& getWeights() const;
    const Matrix& getBiases() const;

    // Setters for weights and biases
    void setWeights(const Matrix& w);
    void setBiases(const Matrix& b);
};

#endif // LAYER_H