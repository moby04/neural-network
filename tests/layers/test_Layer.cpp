#include <gtest/gtest.h>
#include "../../include/layers/Layer.h"
#include "../../include/activations/ActivationFunctions.h"

// Concrete subclass of Layer for testing
class TestLayer : public Layer {
public:
    TestLayer() : Layer(1, 1, std::make_shared<SigmoidActivation>()) {}  // Example constructor

    Matrix forward(const Matrix& input) override {
        // Implement a simple forward pass for testing
        inputCache = input;
        return input;
    }

    Matrix backward(const Matrix& gradOutput) override {
        // Implement a simple backward pass for testing
        return gradOutput;
    }
};

// Test for setting and getting weights
TEST(LayerTest, SetGetWeights) {
    TestLayer layer;
    Matrix weights(1, 1);
    weights.setData({{0.8}});
    layer.setWeights(weights);
    EXPECT_EQ(layer.getWeights().getData()[0][0], 0.8);
}

// Test for setting and getting biases
TEST(LayerTest, SetGetBiases) {
    TestLayer layer;
    Matrix biases(1, 1);
    biases.setData({{0.5}});
    layer.setBiases(biases);
    EXPECT_EQ(layer.getBiases().getData()[0][0], 0.5);
}
