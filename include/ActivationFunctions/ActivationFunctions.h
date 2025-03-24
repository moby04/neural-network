#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include "../Matrix.h"
#include <functional>

/**
 * @brief Abstract base class for all activation functions.
 * Each activation function transforms neuron inputs before passing them to the next layer.
 */
class ActivationFunction {
    public:
        virtual ~ActivationFunction() = default;

        /**
         * @brief Apply the activation function to the input.
         * 
         * This method is used for the forward propagation step.
         * 
         * @param input The input matrix.
         * @return The transformed matrix after applying the activation function.
         */
        virtual Matrix apply(const Matrix& input) const = 0;

        /**
         * @brief Apply the derivative of the activation function to the input.
         * 
         * This method is used for the backward propagation step.
         * 
         * @param input The input matrix.
         * @return The matrix after applying the derivative of the activation function.
         */
        virtual Matrix applyDerivative(const Matrix& input) const = 0;
};

/**
 * @brief Sigmoid Activation Function. I also acts as a base for Swish activation
 * 
 * Formula: σ(x) = 1 / (1 + e^(-x))
 * Derivative: σ'(x) = σ(x) * (1 - σ(x))
 * 
 * - Output range: (0, 1)
 * - Common in logistic regression and simple neural networks.
 * More details: https://en.wikipedia.org/wiki/Sigmoid_function
 */
class SigmoidActivation : public ActivationFunction {
    public:
        Matrix apply(const Matrix& input) const override;
        Matrix applyDerivative(const Matrix& input) const override;
};

/**
 * @brief Swish Activation Function
 * Formula: Swish(x) = x / (1 + e^(-x))
 * Derivative: Swish'(x) = σ(x) + x * σ'(x), where σ(x) is the sigmoid function
 * 
 * - Output range: (-∞, ∞)
 * - Combines the best of ReLU and Sigmoid.
 * - Self-gated variant of Sigmoid.
 * - Helps avoid vanishing gradient problems.
 * More details: https://arxiv.org/abs/1710.05941
 */
class SwishActivation : public SigmoidActivation {
    public:
        Matrix apply(const Matrix& input) const override;
        Matrix applyDerivative(const Matrix& input) const override;
};

/**
 * @brief Base class for all ReLU-based activation functions.
 * This includes standard ReLU and Leaky ReLU.
 */
class ReLUBasedActivation : public ActivationFunction {
    public:
        virtual ~ReLUBasedActivation() = default;
};

/**
 * @brief ReLU (Rectified Linear Unit) Activation Function
 * 
 * Formula: ReLU(x) = max(0, x)
 * Derivative: ReLU'(x) = 1 if x > 0, else 0
 * 
 * - Output range: [0, ∞)
 * - Helps deep networks by avoiding vanishing gradients.
 * - Used extensively in CNNs.
 * More details: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
class ReLUActivation : public ReLUBasedActivation {
    public:
        Matrix apply(const Matrix& input) const override;
        Matrix applyDerivative(const Matrix& input) const override;
};

/**
 * @brief Leaky ReLU Activation Function
 * 
 * Formula: LeakyReLU(x) = x if x > 0, else α * x (where α is small, e.g., 0.01)
 * Derivative: LeakyReLU'(x) = 1 if x > 0, else α
 * 
 * - Output range: (-∞, ∞)
 * - Prevents "dying ReLU" issue where neurons become inactive.
 * - Useful in deeper networks.
 * More details: https://papers.nips.cc/paper_files/paper/2013/hash/7bcec277d0157c7a6ba0d4d7d8c9430e-Abstract.html
 */
class LeakyReLUActivation : public ReLUBasedActivation {
    public:
        Matrix apply(const Matrix& input) const override;
        Matrix applyDerivative(const Matrix& input) const override;
};

/**
 * @brief Base class for all Tanh-based activation functions.
 * This includes standard Tanh and its variants.
 */
class TanhBasedActivation : public ActivationFunction {
    public:
        virtual ~TanhBasedActivation() = default;
};

/**
 * @brief Tanh Activation Function
 * 
 * Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 * Derivative: tanh'(x) = 1 - tanh^2(x)
 * 
 * - Output range: (-1, 1), zero-centered.
 * - Common in recurrent neural networks (RNNs).
 * More details: https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent
 */
class TanhActivation : public TanhBasedActivation {
    public:
        Matrix apply(const Matrix& input) const override;
        Matrix applyDerivative(const Matrix& input) const override;
};

/**
 * @brief Hard Tanh Activation Function (Fast Approximation of Tanh)
 * 
 * Formula: HardTanh(x) = -1 if x < -1, 1 if x > 1, else x
 * Derivative: HardTanh'(x) = 1 for -1 < x < 1, else 0
 * 
 * - Output range: [-1, 1]
 * - Computationally cheaper than regular Tanh.
 * - Used in hardware-efficient AI.
 * More details: https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html
 */
class HardTanhActivation : public TanhBasedActivation {
    public:
        Matrix apply(const Matrix& input) const override;
        Matrix applyDerivative(const Matrix& input) const override;
};

#endif // ACTIVATION_FUNCTIONS_H