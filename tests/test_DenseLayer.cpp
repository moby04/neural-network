#include <gtest/gtest.h>
#include "../include/DenseLayer.h"
#include "../include/ActivationFunctions.h"

// Test DenseLayer Initialization
TEST(DenseLayerTest, Initialization) {
    auto activation = std::make_shared<SigmoidActivation>();
    DenseLayer layer(3, 2, activation);

    EXPECT_EQ(layer.getWeights().getRows(), 2);
    EXPECT_EQ(layer.getWeights().getCols(), 3);
    EXPECT_EQ(layer.getBiases().getRows(), 2);
    EXPECT_EQ(layer.getBiases().getCols(), 1);
}

// Test DenseLayer Forward Propagation with Sigmoid Activation
TEST(DenseLayerTest, ForwardPropagationSigmoid) {
    std::shared_ptr<ActivationFunction> sigmoid = std::make_shared<SigmoidActivation>();
    DenseLayer layer(3, 2, sigmoid);  // 3 inputs, 2 neurons

    Matrix input(3, 1);
    input.setData({{0.5}, {-0.3}, {0.8}});

    Matrix output = layer.forward(input);

    EXPECT_EQ(output.getRows(), 2);
    EXPECT_EQ(output.getCols(), 1);
}

// Test DenseLayer Forward Propagation with ReLU Activation
TEST(DenseLayerTest, ForwardPropagationReLU) {
    std::shared_ptr<ActivationFunction> relu = std::make_shared<ReLUActivation>();
    DenseLayer layer(3, 2, relu);

    Matrix input(3, 1);
    input.setData({{-1.0}, {0.5}, {2.0}});

    Matrix output = layer.forward(input);

    EXPECT_EQ(output.getRows(), 2);
    EXPECT_EQ(output.getCols(), 1);
    EXPECT_GE(output.getData()[0][0], 0.0);  // Ensure ReLU does not produce negative values
}

// Test Setting Custom Weights and Biases
TEST(DenseLayerTest, SetWeightsAndBiases) {
    auto activation = std::make_shared<SigmoidActivation>();
    DenseLayer layer(3, 2, activation);

    Matrix weights(2, 3);
    weights.setData({{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}});
    layer.setWeights(weights);

    Matrix biases(2, 1);
    biases.setData({{0.1f}, {0.2f}});
    layer.setBiases(biases);

    EXPECT_EQ(layer.getWeights(), weights);
    EXPECT_EQ(layer.getBiases(), biases);
}
