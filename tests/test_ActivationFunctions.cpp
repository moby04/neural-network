#include <gtest/gtest.h>
#include "../include/ActivationFunctions.h"

// // Test Sigmoid Activation Function
TEST(ActivationFunctionTest, SigmoidFunction) {
    SigmoidActivation sigmoid;
    Matrix input(2, 2);
    input.setData({{0.0, 1.0}, {-1.0, 2.0}});

    Matrix output = sigmoid.apply(input);

    EXPECT_NEAR(output.getData()[0][0], 0.5, 1e-5);
    EXPECT_NEAR(output.getData()[0][1], 0.731058, 1e-5);
    EXPECT_NEAR(output.getData()[1][0], 0.268941, 1e-5);
    EXPECT_NEAR(output.getData()[1][1], 0.880797, 1e-5);
}

// Test Sigmoid Derivative
TEST(ActivationFunctionTest, SigmoidDerivative) {
    SigmoidActivation sigmoid;
    Matrix input(2, 2);
    input.setData({{0.0, 1.0}, {-1.0, 2.0}});

    Matrix output = sigmoid.applyDerivative(input);

    EXPECT_NEAR(output.getData()[0][0], 0.25, 1e-5);
    EXPECT_NEAR(output.getData()[0][1], 0.196612, 1e-5);
    EXPECT_NEAR(output.getData()[1][0], 0.196612, 1e-5);
    EXPECT_NEAR(output.getData()[1][1], 0.104994, 1e-5);
}

// Test Swish Activation Function
TEST(ActivationFunctionTest, SwishFunction) {
    SwishActivation swish;
    Matrix input(2, 2);
    input.setData({{0.0, 1.0}, {-1.0, 2.0}});

    Matrix output = swish.apply(input);

    EXPECT_NEAR(output.getData()[0][0], 0.0, 1e-5);
    EXPECT_NEAR(output.getData()[0][1], 0.731058, 1e-5);
    EXPECT_NEAR(output.getData()[1][0], -0.268941, 1e-5);
    EXPECT_NEAR(output.getData()[1][1], 1.761594, 1e-5);
}

// Test Swish Derivative
TEST(ActivationFunctionTest, SwishDerivative) {
    SwishActivation swish;
    Matrix input(2, 2);
    input.setData({{0.0, 1.0}, {-1.0, 2.0}});

    Matrix output = swish.applyDerivative(input);

    EXPECT_NEAR(output.getData()[0][0], 0.5, 1e-5);
    EXPECT_NEAR(output.getData()[0][1], 0.927671, 1e-5);
    EXPECT_NEAR(output.getData()[1][0], 0.072329, 1e-5);
    EXPECT_NEAR(output.getData()[1][1], 1.090784, 1e-5);
}

// Test ReLU Activation Function
TEST(ActivationFunctionTest, ReLUFunction) {
    ReLUActivation relu;
    Matrix input(2, 2);
    input.setData({{0.0, -1.0}, {2.0, -3.0}});

    Matrix output = relu.apply(input);

    EXPECT_NEAR(output.getData()[0][0], 0.0, 1e-5);
    EXPECT_NEAR(output.getData()[0][1], 0.0, 1e-5);
    EXPECT_NEAR(output.getData()[1][0], 2.0, 1e-5);
    EXPECT_NEAR(output.getData()[1][1], 0.0, 1e-5);
}

// Test ReLU Derivative
TEST(ActivationFunctionTest, ReLUDerivative) {
    ReLUActivation relu;
    Matrix input(2, 2);
    input.setData({{0.0, -1.0}, {2.0, -3.0}});

    Matrix output = relu.applyDerivative(input);

    EXPECT_NEAR(output.getData()[0][0], 0.0, 1e-5);
    EXPECT_NEAR(output.getData()[0][1], 0.0, 1e-5);
    EXPECT_NEAR(output.getData()[1][0], 1.0, 1e-5);
    EXPECT_NEAR(output.getData()[1][1], 0.0, 1e-5);
}

// Test Leaky ReLU Activation Function
TEST(ActivationFunctionTest, LeakyReLUFunction) {
    LeakyReLUActivation leakyReLU;
    Matrix input(2, 2);
    input.setData({{0.0, -1.0}, {2.0, -3.0}});

    Matrix output = leakyReLU.apply(input);

    EXPECT_NEAR(output.getData()[0][0], 0.0, 1e-5);
    EXPECT_NEAR(output.getData()[0][1], -0.01, 1e-5);
    EXPECT_NEAR(output.getData()[1][0], 2.0, 1e-5);
    EXPECT_NEAR(output.getData()[1][1], -0.03, 1e-5);
}

// Test Leaky ReLU Derivative
TEST(ActivationFunctionTest, LeakyReLUDerivative) {
    LeakyReLUActivation leakyReLU;
    Matrix input(2, 2);
    input.setData({{0.0, -1.0}, {2.0, -3.0}});

    Matrix output = leakyReLU.applyDerivative(input);

    EXPECT_NEAR(output.getData()[0][0], 1.0, 1e-5);
    EXPECT_NEAR(output.getData()[0][1], 0.01, 1e-5);
    EXPECT_NEAR(output.getData()[1][0], 1.0, 1e-5);
    EXPECT_NEAR(output.getData()[1][1], 0.01, 1e-5);
}

// Test Tanh Activation Function
TEST(ActivationFunctionTest, TanhFunction) {
    TanhActivation tanhAct;
    Matrix input(2, 2);
    input.setData({{0.0, 1.0}, {-1.0, 2.0}});

    Matrix output = tanhAct.apply(input);

    EXPECT_NEAR(output.getData()[0][0], 0.0, 1e-5);
    EXPECT_NEAR(output.getData()[0][1], 0.761594, 1e-5);
    EXPECT_NEAR(output.getData()[1][0], -0.761594, 1e-5);
    EXPECT_NEAR(output.getData()[1][1], 0.964027, 1e-5);
}

// Test Tanh Derivative
TEST(ActivationFunctionTest, TanhDerivative) {
    TanhActivation tanhAct;
    Matrix input(2, 2);
    input.setData({{0.0, 1.0}, {-1.0, 2.0}});

    Matrix output = tanhAct.applyDerivative(input);

    EXPECT_NEAR(output.getData()[0][0], 1.0, 1e-5);
    EXPECT_NEAR(output.getData()[0][1], 0.419974, 1e-5);
    EXPECT_NEAR(output.getData()[1][0], 0.419974, 1e-5);
    EXPECT_NEAR(output.getData()[1][1], 0.0706508, 1e-5);
}

// Test Hard Tanh Activation Function
TEST(ActivationFunctionTest, HardTanhFunction) {
    HardTanhActivation hardTanh;
    Matrix input(2, 2);
    input.setData({{-2.0, 0.5}, {1.5, -0.8}});

    Matrix output = hardTanh.apply(input);

    EXPECT_NEAR(output.getData()[0][0], -1.0, 1e-5);
    EXPECT_NEAR(output.getData()[0][1], 0.5, 1e-5);
    EXPECT_NEAR(output.getData()[1][0], 1.0, 1e-5);
    EXPECT_NEAR(output.getData()[1][1], -0.8, 1e-5);
}

// Test Hard Tanh Derivative
TEST(ActivationFunctionTest, HardTanhDerivative) {
    HardTanhActivation hardTanh;
    Matrix input(2, 2);
    input.setData({{-2.0, 0.5}, {1.5, -0.8}});

    Matrix output = hardTanh.applyDerivative(input);

    EXPECT_NEAR(output.getData()[0][0], 0.0, 1e-5);
    EXPECT_NEAR(output.getData()[0][1], 1.0, 1e-5);
    EXPECT_NEAR(output.getData()[1][0], 0.0, 1e-5);
    EXPECT_NEAR(output.getData()[1][1], 1.0, 1e-5);
}
