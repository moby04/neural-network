#include <gtest/gtest.h>
#include "../../include/Layer/StatefulLayer.h"
#include "../../include/ActivationFunctions/ActivationFunctions.h"

// Concrete subclass of StatefulLayer for testing
class TestStatefulLayer : public StatefulLayer {
public:
    TestStatefulLayer() : StatefulLayer(1, 1, std::make_shared<SigmoidActivation>()) {}  // Example constructor

    Matrix forward(const Matrix& input) override {
        // Implement a simple forward pass for testing
        inputCache = input;
        return input;
    }

    Matrix backward(const Matrix& gradOutput) override {
        // Implement a simple backward pass for testing
        return gradOutput;
    }

    void resetStates() override {
        // empty function just to make the class instantiable
    }
};

// Test for clearing input cache
TEST(StatefulLayerTest, ClearInputCache) {
    TestStatefulLayer layer;
    Matrix input(1, 1);
    input.setData({{1.0}});
    layer.forward(input);
    layer.clearInputCache();
    EXPECT_TRUE(layer.getInputCache().isEmpty(true));
}

// Test for getting input cache
TEST(StatefulLayerTest, GetInputCache) {
    TestStatefulLayer layer;
    Matrix input(1, 1);
    input.setData({{1.0}});
    layer.forward(input);
    EXPECT_EQ(layer.getInputCache().getData()[0][0], 1.0);
}
