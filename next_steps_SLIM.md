# C++ Matrix Simulation - Slim Implementation Plan

## **Goal: Ready Demo Version in 2-3 Days**
This plan focuses on implementing a **minimal training loop** with **basic optimizers, loss functions, and a simple dataset** to quickly produce a working simulation.

---

## **Phase 1: Core Training Loop (Day 1)**

### **1. Implement Basic Optimizer (SGD)**
- Create a simple `SGDOptimizer` class with:
  - Learning rate (`η`) adjustment.
  - Weight update rule:
    ```
    W = W - η * gradient
    ```
- Modify `DenseLayer::backward` to apply weight updates.
- Test with random weights and gradients.

**Estimated Time:** 2-3 hours

---

### **2. Add a Simple Loss Function (MSE)**
- Implement `MeanSquaredError` with:
  - Loss calculation:
    ```
    Loss = (1/n) * Σ (y_true - y_pred)²
    ```
  - Gradient:
    ```
    dL/dy_pred = 2 * (y_pred - y_true) / n
    ```
- Modify `NeuralNetwork` to use this loss function.
- Test with sample predictions.

**Estimated Time:** 1-2 hours

---

## **Phase 2: Training and Simulation (Day 2)**

### **3. Implement Basic Training Loop**
- Update `NeuralNetwork::train` to:
  - Perform **forward pass**.
  - Compute **loss and gradients**.
  - Apply **SGD updates**.
- Add simple logging (loss per epoch).

**Estimated Time:** 3 hours

---

### **4. Create a Simple Dataset for Simulation**
- Generate a **toy dataset** (e.g., predicting a sine wave or a linear function).
- Normalize data if needed.
- Split into **training** and **validation** sets.

**Estimated Time:** 1 hour

---

## **Phase 3: Evaluation and Debugging (Day 3)**

### **5. Implement Basic Model Evaluation**
- Compute **final loss** on test data.
- Add **accuracy measurement** (for classification tasks).
- Print and visualize results.

**Estimated Time:** 2 hours

---

### **6. Add Simple Visualization (Optional)**
- Export results for **plotting**.
- Show **training loss** over epochs.

**Estimated Time:** 1-2 hours  

---

## **Final Timeline (2-3 Days)**

| Day | Task | Estimated Time |
|-----|------|---------------|
| 1 | Implement SGD optimizer | 2-3 hrs |
| 1 | Add MSE loss function | 1-2 hrs |
| 2 | Implement training loop | 3 hrs |
| 2 | Create toy dataset | 1 hr |
| 3 | Evaluate model performance | 2 hrs |
| 3 | Add visualization (optional) | 1-2 hrs |

---

## **Next Steps After Demo**
Once the **demo works**, we can:
- Add **better optimizers** (Adam, RMSprop).
- Implement **batch processing**.
- Expand to **sequence handling (RNN/LSTM)**.
- Optimize **matrix operations** for performance.

---

This plan ensures a **working prototype in 2-3 days** while keeping the structure extendable for future improvements.
