#include <gtest/gtest.h>
#include "../../include/Layer/RNNLayer.h"

// Test RNN Forward Pass
TEST(RNNLayerTest, ForwardPass) {
    RNNLayer rnn(3, 2);  // Input size: 3, Hidden size: 2
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});

    Matrix output = rnn.forward(input);

    EXPECT_EQ(output.getRows(), 1);
    EXPECT_EQ(output.getCols(), 2);
    // We can't check exact values since weights are random, but we ensure they exist
    EXPECT_FALSE(output.isEmpty());
}

TEST(RNNLayerTest, InputCacheUpdatedInForward) {
    RNNLayer rnn(3, 2);  // Input size: 3, Hidden size: 2
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});

    rnn.forward(input);

    // Access inputCache from Layer class (assuming protected or a getter method exists)
    Matrix cachedInput = rnn.getInputCache(); 

    EXPECT_EQ(cachedInput.getRows(), 1);
    EXPECT_EQ(cachedInput.getCols(), 3);
    EXPECT_EQ(cachedInput.getData()[0][0], 1.0);
    EXPECT_EQ(cachedInput.getData()[0][1], 0.5);
    EXPECT_EQ(cachedInput.getData()[0][2], -0.5);
}

// Test RNN Backward Pass
TEST(RNNLayerTest, BackwardPass) {
    RNNLayer rnn(3, 2);
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});
    
    Matrix gradOutput(1, 2);
    gradOutput.setData({{0.1, -0.2}});

    rnn.forward(input); // Must call forward first to store hidden state
    Matrix gradInput = rnn.backward(gradOutput);

    EXPECT_EQ(gradInput.getRows(), 1);
    EXPECT_EQ(gradInput.getCols(), 3);
    EXPECT_FALSE(gradInput.isEmpty());
}

// Test Hidden State Reset
TEST(RNNLayerTest, DifferentInputAfterReset) {
    RNNLayer rnn(3, 2);
    
    Matrix input1(1, 3);
    input1.setData({{1.0, 0.5, -0.5}});

    Matrix input2(1, 3);
    input2.setData({{-0.2, 0.8, 1.5}});  // Different input after reset

    Matrix output1 = rnn.forward(input1);
    rnn.resetStates();
    Matrix output2 = rnn.forward(input2);

    EXPECT_NE(output1, output2);
}

TEST(RNNLayerTest, ResetHiddenState) {
    RNNLayer rnn(3, 2);
    
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});

    rnn.forward(input);
    Matrix hiddenBeforeReset = rnn.getHiddenState(); 

    Matrix zeroMatrix(hiddenBeforeReset.getRows(), hiddenBeforeReset.getCols());
    zeroMatrix.setData(0.0);

    // verify hiddenState is not zero before reset
    EXPECT_NE(hiddenBeforeReset, zeroMatrix);

    rnn.resetStates();
    Matrix hiddenAfterReset = rnn.getHiddenState(); 

    // ensure hiddenState is now zero after reset
    EXPECT_EQ(hiddenAfterReset, zeroMatrix);
}