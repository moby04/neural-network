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

/**
 * @brief Matrix class for handling matrix operations.
 * 
 * This class provides the basic functionalities needed to implement layers in a neural network.
 * It supports various matrix operations such as addition, subtraction, multiplication, and transposition.
 */
class Matrix {
private:
    std::string name;
    size_t rows;
    size_t cols;
    std::vector<std::vector<double>> data; 

public:
    // Constructors and Destructor
    /**
     * @brief Construct a new Matrix object.
     * 
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param name Name of the matrix (optional).
     */
    Matrix(size_t rows, size_t cols, const std::string& name = "UNNAMED");

    /**
     * @brief Copy constructor.
     * 
     * @param other The matrix to copy.
     */
    Matrix(const Matrix& other); // Copy constructor

    /**
     * @brief Destroy the Matrix object.
     */
    ~Matrix();

    // Getters
    size_t getRows() const;
    size_t getCols() const;
    const std::vector<std::vector<double>>& getData() const;
    std::span<const double> getRow(size_t row) const;
    std::span<const double> getCol(size_t col) const;
    std::string getName() const;

    // Setters
    /**
     * @brief Set the name of the matrix.
     * 
     * @param name The new name.
     */
    void setName(const std::string& name);

    /**
     * @brief Set the data of the matrix.
     * 
     * @param newData The new data.
     * @return A reference to the matrix.
     */
    Matrix& setData(const std::vector<std::vector<double>>& newData);

    /**
     * @brief Set all elements of the matrix to a specific value.
     * 
     * @param value The value to set.
     * @return A reference to the matrix.
     */
    Matrix& setData(double value);

    // Utility Methods
    /**
     * @brief Fill the matrix by hand (from console input).
     */
    void fillByHand();

    /**
     * @brief Print the matrix to the console.
     * 
     * This method prints the matrix along with its name to the console.
     */
    void print() const;

    /**
     * @brief Randomize the matrix elements within a given range.
     * 
     * @param min Minimum value.
     * @param max Maximum value.
     */
    void randomize(double min = 0.0, double max = 1.0);

    /**
     * @brief Apply a function to each element of the matrix.
     * 
     * @param func The function to apply.
     * @return A new matrix with the function applied.
     */
    Matrix applyFunction(const std::function<double(double)>& func) const;

    /**
     * @brief Create an identity matrix.
     * 
     * @param size The size of the identity matrix.
     * @param name The name of the matrix (optional).
     * @return The identity matrix.
     */
    static Matrix createIdentityMatrix(size_t size, const std::string& name = "UNNAMED");

    /**
     * @brief Check if the matrix is empty.
     * 
     * This method checks if the matrix is empty. If `checkForNonZeroData` is true, it also checks if the matrix contains only zeroes.
     * 
     * @param checkForNonZeroData If true, check if the matrix contains only zeroes.
     * @return True if the matrix is empty, false otherwise.
     */
    bool isEmpty(bool checkForNonZeroData = false) const;

    // Matrix Operations
    /**
     * @brief Add two matrices.
     * 
     * @param other The matrix to add.
     * @return The result of the addition.
     */
    Matrix add(const Matrix& other) const;

    /**
     * @brief Subtract one matrix from another.
     * 
     * @param other The matrix to subtract.
     * @return The result of the subtraction.
     */
    Matrix subtract(const Matrix& other) const;

    /**
     * @brief Multiply two matrices.
     * 
     * @param other The matrix to multiply with.
     * @param elementWise If true, perform element-wise multiplication; otherwise, perform matrix multiplication.
     * @return The result of the multiplication.
     */
    Matrix multiply(const Matrix& other, bool elementWise = true) const;

    /**
     * @brief Multiply the matrix by a scalar.
     * 
     * @param scalar The scalar value.
     * @return The result of the multiplication.
     */
    Matrix multiply(double scalar) const;

    /**
     * @brief Transpose the matrix.
     * 
     * @return The transposed matrix.
     */
    Matrix transpose() const;

    /**
     * @brief Sum the rows of the matrix.
     * 
     * @return A column vector with the sums.
     */
    Matrix sumRows() const;    // Sums across rows, returns column vector

    /**
     * @brief Sum the columns of the matrix.
     * 
     * @return A row vector with the sums.
     */
    Matrix sumColumns() const;

    // Overloaded Operators
    /**
     * @brief Add two matrices using the + operator.
     * 
     * @param other The matrix to add.
     * @return The result of the addition.
     */
    Matrix operator+(const Matrix& other) const;

    /**
     * @brief Subtract one matrix from another using the - operator.
     * 
     * @param other The matrix to subtract.
     * @return The result of the subtraction.
     */
    Matrix operator-(const Matrix& other) const;

    /**
     * @brief Multiply two matrices element-wise using the * operator.
     * 
     * Note: This performs element-wise multiplication, not matrix multiplication.
     * 
     * @param other The matrix to multiply with.
     * @return The result of the multiplication.
     */
    Matrix operator*(const Matrix& other) const;

    /**
     * @brief Multiply the matrix by a scalar using the * operator.
     * 
     * @param scalar The scalar value.
     * @return The result of the multiplication.
     */
    Matrix operator*(double scalar) const;

    /**
     * @brief Divide the matrix by a scalar using the / operator.
     * 
     * @param scalar The scalar value.
     * @return The result of the division.
     */
    Matrix operator/(double scalar) const;

    /**
     * @brief Access an element of the matrix.
     * 
     * @param row The row index.
     * @param col The column index.
     * @return A reference to the element.
     */
    double& operator()(size_t row, size_t col);

    /**
     * @brief Access an element of the matrix (const version).
     * 
     * @param row The row index.
     * @param col The column index.
     * @return A const reference to the element.
     */
    const double& operator()(size_t row, size_t col) const;

    /**
     * @brief Compare two matrices for equality.
     * 
     * @param other The matrix to compare with.
     * @return True if the matrices are equal, false otherwise.
     */
    bool operator==(const Matrix& other) const;

    /**
     * @brief Compare two matrices using the three-way comparison operator.
     * 
     * @param other The matrix to compare with.
     * @return The result of the comparison.
     */
    std::partial_ordering operator<=>(const Matrix& other) const; 

    /**
     * @brief Custom comparison function with tolerance.
     * 
     * @param other The matrix to compare with.
     * @param tolerance The tolerance for comparison.
     * @return True if the matrices are equal within the given tolerance, false otherwise.
     */
    bool isEqual(const Matrix& other, double tolerance = 1e-5) const;

    // Friend functions for overloading the << and >> operators
    /**
     * @brief Output the matrix to a stream.
     * 
     * This operator outputs the matrix elements to a stream in a formatted manner.
     * 
     * @param os The output stream.
     * @param matrix The matrix to output.
     * @return The output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);

    /**
     * @brief Input the matrix from a stream.
     * 
     * This operator inputs the matrix elements from a stream.
     * 
     * @param is The input stream.
     * @param matrix The matrix to input.
     * @return The input stream.
     */
    friend std::istream& operator>>(std::istream& is, Matrix& matrix);
};

#endif // MATRIX_H
