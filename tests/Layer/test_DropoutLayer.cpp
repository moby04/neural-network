#include <gtest/gtest.h>
#include "../../include/Layer/DropoutLayer.h"

// Test Dropout Layer does not change input shape
TEST(DropoutLayerTest, OutputShapeUnchanged) {
    auto activation = std::make_shared<SigmoidActivation>();
    DropoutLayer layer(3, 3, activation, 0.5); // 3 neurons, 50% dropout
    Matrix input(3, 1);
    input.setData({{1.0}, {2.0}, {3.0}});

    Matrix output = layer.forward(input);

    EXPECT_EQ(output.getRows(), 3);
    EXPECT_EQ(output.getCols(), 1);
}

// Test Dropout Effect (Ensure some elements are zero)
TEST(DropoutLayerTest, SomeNeuronsAreDropped) {
    auto activation = std::make_shared<SigmoidActivation>();
    DropoutLayer layer(5, 5, activation, 0.5); // 50% dropout probability
    Matrix input(5, 1);
    input.setData({{1.0}, {1.0}, {1.0}, {1.0}, {1.0}});

    Matrix output = layer.forward(input);

    int zeroCount = 0;
    for (size_t i = 0; i < output.getRows(); ++i) {
        if (output(i, 0) == 0.0) {
            zeroCount++;
        }
    }
    EXPECT_GT(zeroCount, 0);  // At least one neuron should be dropped
}

// Test Dropout with 0% Rate (No Neurons Dropped)
TEST(DropoutLayerTest, ZeroDropout) {
    auto activation = std::make_shared<SigmoidActivation>();
    DropoutLayer layer(3, 3, activation, 0.0); // No dropout
    Matrix input(3, 1);
    input.setData({{1.0}, {2.0}, {3.0}});

    Matrix output = layer.forward(input);

    EXPECT_EQ(output, input);  // Output should be identical to input
}

// Test Dropout with 100% Rate (All Neurons Dropped)
TEST(DropoutLayerTest, FullDropout) {
    auto activation = std::make_shared<SigmoidActivation>();
    DropoutLayer layer(3, 3, activation, 1.0); // 100% dropout (all neurons dropped)
    Matrix input(3, 1);
    input.setData({{1.0}, {2.0}, {3.0}});

    Matrix output = layer.forward(input);

    for (size_t i = 0; i < output.getRows(); ++i) {
        EXPECT_EQ(output(i, 0), 0.0);  // All values should be zero
    }
}
