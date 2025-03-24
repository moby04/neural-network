#ifndef LAYER_H
#define LAYER_H

#include "../ActivationFunctions/ActivationFunctions.h"
#include "../Matrix.h"
#include <functional>
#include <memory>

/**
 * @brief Abstract base class for all neural network layers.
 * 
 * This class defines the basic structure and functionalities of a neural network layer.
 * Each layer has weights, biases, and an activation function.
 */
class Layer {
protected:
    Matrix weights;
    Matrix biases;
    std::shared_ptr<ActivationFunction> activation;
    Matrix inputCache;

public:
    // Constructor and Destructor
    Layer(size_t inputSize, size_t neurons, std::shared_ptr<ActivationFunction> activationFunc)
        : weights(neurons, inputSize), biases(neurons, 1), activation(activationFunc), inputCache(inputSize, 1)  {}
    virtual ~Layer() = default;

    // Forward and Backward Propagation
    /**
     * @brief Perform the forward pass through the layer.
     * 
     * This method takes an input matrix and returns the output matrix after applying the layer's weights, biases, and activation function.
     * 
     * @param input The input matrix.
     * @return The output matrix.
     */
    virtual Matrix forward(const Matrix& input) = 0;

    /**
     * @brief Perform the backward pass through the layer.
     * 
     * This method takes the gradient of the loss with respect to the layer's output and returns the gradient of the loss with respect to the layer's input.
     * 
     * @param gradient The gradient of the loss with respect to the layer's output.
     * @return The gradient of the loss with respect to the layer's input.
     */
    virtual Matrix backward(const Matrix& gradient) = 0; 

    // Getters for weights and biases
    const Matrix& getWeights() const;
    const Matrix& getBiases() const;
    const Matrix& getInputCache() const;

    // Setters for weights and biases
    void setWeights(const Matrix& w);
    void setBiases(const Matrix& b);
};

#endif // LAYER_H