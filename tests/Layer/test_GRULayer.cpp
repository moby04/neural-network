#include <gtest/gtest.h>
#include "../../include/Layer/GRULayer.h"

// **1. Test Forward Pass**
TEST(GRULayerTest, ForwardPass) {
    GRULayer gru(3, 2);  // Input size: 3, Hidden size: 2
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});

    Matrix output = gru.forward(input);

    EXPECT_EQ(output.getRows(), 1);
    EXPECT_EQ(output.getCols(), 2);
    EXPECT_FALSE(output.isEmpty());
}

// **2. Test Backward Pass**
TEST(GRULayerTest, BackwardPass) {
    GRULayer gru(3, 2);
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});

    Matrix gradOutput(1, 2);
    gradOutput.setData({{0.1, -0.2}});

    gru.forward(input);
    Matrix gradInput = gru.backward(gradOutput);

    EXPECT_EQ(gradInput.getRows(), 1);
    EXPECT_EQ(gradInput.getCols(), 3);
    EXPECT_FALSE(gradInput.isEmpty());
}

// **3. Edge Case: Empty Input Should Throw Exception**
TEST(GRULayerTest, ForwardWithEmptyInput) {
    GRULayer gru(3, 2);
    Matrix input(0, 3);  // Empty input

    EXPECT_THROW(gru.forward(input), std::runtime_error);
}

// **4. Edge Case: Backward Before Forward Should Throw**
TEST(GRULayerTest, BackwardBeforeForward) {
    GRULayer gru(3, 2);
    Matrix gradOutput(1, 2);
    gradOutput.setData({{0.1, -0.2}});

    EXPECT_THROW(gru.backward(gradOutput), std::runtime_error);
}

// **5. Test Reset State**
TEST(GRULayerTest, ResetState) {
    GRULayer gru(3, 2);
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});

    gru.forward(input);
    Matrix hiddenBeforeReset = gru.getHiddenState();

    gru.resetStates();
    Matrix hiddenAfterReset = gru.getHiddenState();

    Matrix zeroHidden(hiddenBeforeReset.getRows(), hiddenBeforeReset.getCols());
    zeroHidden.setData(0.0);

    EXPECT_NE(hiddenBeforeReset, zeroHidden);
    EXPECT_EQ(hiddenAfterReset, zeroHidden);
}

// **6. Ensure Reset Works Multiple Times**
TEST(GRULayerTest, MultipleResets) {
    GRULayer gru(3, 2);
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});

    gru.forward(input);
    gru.resetStates();
    gru.resetStates();  // Call reset again

    Matrix zeroHidden(1, 2);
    zeroHidden.setData(0.0);

    EXPECT_EQ(gru.getHiddenState(), zeroHidden);
}

// **7. Test Get Hidden State**
TEST(GRULayerTest, GetHiddenState) {
    GRULayer gru(3, 2);
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});

    gru.forward(input);
    Matrix hidden = gru.getHiddenState();

    EXPECT_EQ(hidden.getRows(), 1);
    EXPECT_EQ(hidden.getCols(), 2);
    EXPECT_FALSE(hidden.isEmpty());
}
