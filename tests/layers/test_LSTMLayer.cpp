#include <gtest/gtest.h>
#include "../../include/layers/LSTMLayer.h"

// Test Forward Pass with Normal Input
TEST(LSTMLayerTest, ForwardPass) {
    LSTMLayer lstm(3, 2);
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});

    Matrix output = lstm.forward(input);

    EXPECT_EQ(output.getRows(), 1);
    EXPECT_EQ(output.getCols(), 2);
    EXPECT_FALSE(output.isEmpty());
}

// Test Backward Pass Normally
TEST(LSTMLayerTest, BackwardPass) {
    LSTMLayer lstm(3, 2);  // Input size: 3, Hidden size: 2
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});
    
    Matrix gradOutput(1, 2);  // Expected: (batchSize x hiddenSize)
    gradOutput.setData({{0.1, -0.2}});

    lstm.forward(input);
    Matrix gradInput = lstm.backward(gradOutput);

    EXPECT_EQ(gradInput.getRows(), 1);
    EXPECT_EQ(gradInput.getCols(), 3);
    EXPECT_FALSE(gradInput.isEmpty());
}

// Edge Case: Empty Input (Should Throw Exception)
TEST(LSTMLayerTest, ForwardWithEmptyInput) {
    LSTMLayer lstm(3, 2);
    Matrix input(0, 3);  // Empty input

    EXPECT_THROW(lstm.forward(input), std::runtime_error);
}

// Edge Case: Calling Backward Before Forward (Should Throw)
TEST(LSTMLayerTest, BackwardBeforeForward) {
    LSTMLayer lstm(3, 2);  // Ensure a fresh instance
    Matrix gradOutput(1, 2);
    gradOutput.setData({{0.1, -0.2}});

    EXPECT_THROW(lstm.backward(gradOutput), std::runtime_error);
}

// Test Resetting Hidden & Cell States
TEST(LSTMLayerTest, ResetStates) {
    LSTMLayer lstm(3, 2);
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});

    lstm.forward(input);

    Matrix hiddenBeforeReset = lstm.getHiddenState();
    Matrix cellBeforeReset = lstm.getCellState();

    lstm.resetStates();
    
    Matrix hiddenAfterReset = lstm.getHiddenState();
    Matrix cellAfterReset = lstm.getCellState();

    Matrix zeroHidden(hiddenBeforeReset.getRows(), hiddenBeforeReset.getCols());
    zeroHidden.setData(0.0);

    Matrix zeroCell(cellBeforeReset.getRows(), cellBeforeReset.getCols());
    zeroCell.setData(0.0);

    EXPECT_NE(hiddenBeforeReset, zeroHidden);
    EXPECT_NE(cellBeforeReset, zeroCell);
    EXPECT_EQ(hiddenAfterReset, zeroHidden);
    EXPECT_EQ(cellAfterReset, zeroCell);
}

// Ensure Reset Works Multiple Times
TEST(LSTMLayerTest, MultipleResets) {
    LSTMLayer lstm(3, 2);
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});

    lstm.forward(input);
    lstm.resetStates();
    lstm.resetStates();  // Call reset again

    Matrix zeroHidden(1, 2);
    zeroHidden.setData(0.0);
    Matrix zeroCell(1, 2);
    zeroCell.setData(0.0);

    EXPECT_EQ(lstm.getHiddenState(), zeroHidden);
    EXPECT_EQ(lstm.getCellState(), zeroCell);
}

// Test Getters for Hidden & Cell State
TEST(LSTMLayerTest, GetHiddenAndCellState) {
    LSTMLayer lstm(3, 2);
    Matrix input(1, 3);
    input.setData({{1.0, 0.5, -0.5}});

    lstm.forward(input);
    Matrix hidden = lstm.getHiddenState();
    Matrix cell = lstm.getCellState();

    EXPECT_EQ(hidden.getRows(), 1);
    EXPECT_EQ(hidden.getCols(), 2);
    EXPECT_FALSE(hidden.isEmpty());

    EXPECT_EQ(cell.getRows(), 1);
    EXPECT_EQ(cell.getCols(), 2);
    EXPECT_FALSE(cell.isEmpty());
}
