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
    std::vector<std::vector<double>> data; 

public:
    Matrix(size_t rows, size_t cols, const std::string& name = "UNNAMED");
    Matrix(const Matrix& other); // Copy constructor
    ~Matrix();

    void setName(const std::string& name);
    std::string getName() const;

    void fillByHand();
    void print() const;
    void randomize(double min = 0.0, double max = 1.0);

    Matrix applyFunction(const std::function<double(double)>& func) const;

    static Matrix createIdentityMatrix(size_t size, const std::string& name = "UNNAMED");

    // Matrix operations
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix multiply(const Matrix& other, bool elementWise = true) const;
    Matrix multiply(double scalar) const;
    Matrix transpose() const;

    // Getters
    size_t getRows() const;
    size_t getCols() const;
    const std::vector<std::vector<double>>& getData() const;
    std::span<const double> getRow(size_t row) const;
    std::span<const double> getCol(size_t col) const;

    // Setters
    void setData(const std::vector<std::vector<double>>& newData);
    void setData(double value);

    // Friend functions for overloading the << and >> operators
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
    friend std::istream& operator>>(std::istream& is, Matrix& matrix);

    // Overloaded operators
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;

    double& operator()(size_t row, size_t col);
    const double& operator()(size_t row, size_t col) const;

    bool operator==(const Matrix& other) const;
    std::partial_ordering operator<=>(const Matrix& other) const; 

    // Custom comparison function
    bool isEqual(const Matrix& other, double tolerance = 1e-5) const;
};

#endif // MATRIX_H
