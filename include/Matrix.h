#ifndef MATRIX_H
#define MATRIX_H

#include <cmath>
#include <compare>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

class Matrix {
private:
    std::string name;
    size_t rows;
    size_t cols;
    // Vector handles the memory management so it's better for handling data owned by the object.
    // on the other hand span<> used below is a non-owning reference that is faster and more efficient
    // so it's better for passing data to and from functions.
    std::vector<std::vector<float>> data; 

public:
    Matrix(size_t rows, size_t cols, const std::string& name = "UNNAMED");
    Matrix(const Matrix& other); // Copy constructor
    ~Matrix();

    void setName(const std::string& name);
    std::string getName() const;

    void fillByHand();
    void print() const;
    void randomize(float min = 0.0f, float max = 1.0f);

    Matrix applyFunction(const std::function<float(float)>& func) const;

    static Matrix createIdentityMatrix(size_t size, const std::string& name = "UNNAMED");

    // Matrix operations
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix multiply(const Matrix& other, bool elementWise = true) const;
    Matrix multiply(float scalar) const;
    Matrix transpose() const;

    // Getters
    size_t getRows() const;
    size_t getCols() const;
    const std::vector<std::vector<float>>& getData() const;
    std::span<const float> getRow(size_t row) const;
    std::span<const float> getCol(size_t col) const;

    // Setters
    void setData(const std::vector<std::vector<float>>& newData);
    void setData(float value);

    // Friend functions for overloading the << and >> operators
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
    friend std::istream& operator>>(std::istream& is, Matrix& matrix);

    // Overloaded operators
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(float scalar) const;

    float& operator()(size_t row, size_t col);
    const float& operator()(size_t row, size_t col) const;

    bool operator==(const Matrix& other) const;
    std::partial_ordering operator<=>(const Matrix& other) const;

    // Custom comparison function
    bool isEqual(const Matrix& other, float tolerance = 1e-5) const;
};

#endif // MATRIX_H
