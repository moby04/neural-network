#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include "Matrix.h"
#include <functional>

/**
 * @brief Abstract base class for all activation functions.
 * Each activation function transforms neuron inputs before passing them to the next layer.
 */
class ActivationFunction {
    public:
        virtual ~ActivationFunction() = default;
        virtual Matrix apply(const Matrix& input) const = 0;
};

/**
 * @brief Base class for all Sigmoid-based activation functions.
 * This includes variations like Swish and standard Sigmoid.
 */
class SigmoidBasedActivation : public ActivationFunction {
    public:
        virtual ~SigmoidBasedActivation() = default;
};

/**
 * @brief Sigmoid Activation Function
 * 
 * Formula: σ(x) = 1 / (1 + e^(-x))
 * - Output range: (0, 1)
 * - Common in logistic regression and simple neural networks.
 * More details: https://en.wikipedia.org/wiki/Sigmoid_function
 */
class SigmoidActivation : public SigmoidBasedActivation {
    public:
        Matrix apply(const Matrix& input) const override;
};

/**
 * @brief Swish Activation Function
 * 
 * Formula: Swish(x) = x / (1 + e^(-x))
 * - Self-gated variant of Sigmoid.
 * - Helps avoid vanishing gradient problems.
 * More details: https://arxiv.org/abs/1710.05941
 */
class SwishActivation : public SigmoidBasedActivation {
    public:
        Matrix apply(const Matrix& input) const override;
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
 * - Helps deep networks by avoiding vanishing gradients.
 * - Used extensively in CNNs.
 * More details: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
 */
class ReLUActivation : public ReLUBasedActivation {
    public:
        Matrix apply(const Matrix& input) const override;
};

/**
 * @brief Leaky ReLU Activation Function
 * 
 * Formula: LeakyReLU(x) = x if x > 0, else α * x (where α is small)
 * - Prevents "dying ReLU" issue where neurons become inactive.
 * - Useful in deeper networks.
 * More details: https://papers.nips.cc/paper_files/paper/2013/hash/7bcec277d0157c7a6ba0d4d7d8c9430e-Abstract.html
 */
class LeakyReLUActivation : public ReLUBasedActivation {
    public:
        Matrix apply(const Matrix& input) const override;
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
 * - Output range: (-1, 1), zero-centered.
 * - Common in recurrent neural networks (RNNs).
 * More details: https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent
 */
class TanhActivation : public TanhBasedActivation {
    public:
        Matrix apply(const Matrix& input) const override;
};

/**
 * @brief Hard Tanh Activation Function (Fast Approximation of Tanh)
 * 
 * Formula: HardTanh(x) = -1 if x < -1, 1 if x > 1, else x
 * - Computationally cheaper than regular Tanh.
 * - Used in hardware-efficient AI.
 * More details: https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html
 */
class HardTanhActivation : public TanhBasedActivation {
    public:
        Matrix apply(const Matrix& input) const override;
};

#endif // ACTIVATION_FUNCTIONS_H