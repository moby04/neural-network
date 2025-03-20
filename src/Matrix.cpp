#include "../include/Matrix.h"

Matrix::Matrix(size_t rows, size_t cols, const std::string& name)
    : rows(rows), cols(cols), name(name), data(rows, std::vector<double>(cols, 0.0)) {}

// Copy constructor
Matrix::Matrix(const Matrix& other)
    : name(other.name), rows(other.rows), cols(other.cols), data(other.data) {}

Matrix::~Matrix() {}

void Matrix::setName(const std::string& name) {
    this->name = name;
}

std::string Matrix::getName() const {
    return name;
}

size_t Matrix::getRows() const {
    return rows;
}

size_t Matrix::getCols() const {
    return cols;
}

const std::vector<std::vector<double>>& Matrix::getData() const {
    return data;
}

std::span<const double> Matrix::getRow(size_t row) const {
    if (row >= rows) {
        throw std::out_of_range("Row index out of range.");
    }
    return std::span<const double>(data[row]);
}

std::span<const double> Matrix::getCol(size_t col) const {
    if (col >= cols) {
        throw std::out_of_range("Column index out of range.");
    }
    std::vector<double> colData(rows);
    for (size_t i = 0; i < rows; ++i) {
        colData[i] = data[i][col];
    }
    return std::span<const double>(colData);
}

Matrix& Matrix::setData(const std::vector<std::vector<double>>& newData) {
    if (newData.empty()) {
        throw std::invalid_argument("Data cannot be empty.");
    }
    size_t newCols = newData[0].size();
    for (const auto& row : newData) {
        if (row.size() != newCols) {
            throw std::invalid_argument("All rows must have the same number of columns.");
        }
    }
    data = newData;
    rows = newData.size();
    cols = newCols;

    return *this;
}

Matrix& Matrix::setData(double value) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = value;
        }
    }
    return *this;
}

bool Matrix::isEmpty(bool checkForNonZeroData ) const {
    if (rows == 0 || cols == 0 || data.empty()) {
        return true;
    }
    if (checkForNonZeroData) {
        for (const auto& row : data) {
            for (double value : row) {
                if (value != 0.0) {
                    return false;  // Matrix has meaningful data
                }
            }
        }
        return true;  // Only zero values are present
    }
    return false;  // Matrix exists with allocated space
}

Matrix Matrix::sumRows() const {
    if (rows == 0 || cols == 0 || data.empty()) {
        throw std::runtime_error("Cannot sum rows of an empty matrix.");
    }

    Matrix result(rows, 1, "sumRows");
    for (size_t i = 0; i < rows; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < cols; j++) {
            sum += data[i][j];
        }
        result(i, 0) = sum;
    }
    return result;
}

Matrix Matrix::sumColumns() const {
    if (rows == 0 || cols == 0 || data.empty()) {
        throw std::runtime_error("Cannot sum rows of an empty matrix.");
    }

    Matrix result(1, cols, "sumColumns");
    for (size_t j = 0; j < cols; j++) {
        double sum = 0.0;
        for (size_t i = 0; i < rows; i++) {
            sum += data[i][j];
        }
        result(0, j) = sum;
    }
    return result;
}

void Matrix::fillByHand() {
    std::cin >> *this;
}

void Matrix::print() const {
    std::cout << "Matrix name: " << name << "\n";
    std::cout << *this;
}

Matrix Matrix::createIdentityMatrix(size_t size, const std::string& name) {
    Matrix identity(size, size, name);
    for (size_t i = 0; i < size; ++i) {
        identity.data[i][i] = 1.0;
    }
    return identity;
}

Matrix Matrix::add(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }
    Matrix result(rows, cols, "Result");
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices must have the same dimensions for subtraction.");
    }
    Matrix result(rows, cols, "Result");
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::multiply(const Matrix& other, bool elementWise) const {
    if (elementWise) {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrices must have the same dimensions for element-wise multiplication.");
        }
        Matrix result(rows, cols, "Result");
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    } else {
        if (cols != other.rows) {
            throw std::invalid_argument("Matrices have incompatible sizes for multiplication.");
        }
        Matrix result(rows, other.cols, "Result");
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                result.data[i][j] = 0;
                for (size_t k = 0; k < cols; ++k) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }
}

Matrix Matrix::multiply(double scalar) const {
    Matrix result(rows, cols, "Result");
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = data[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows, "Transposed");
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[j][i] = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::applyFunction(const std::function<double(double)>& func) const {
    Matrix result(rows, cols, "Result");
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.data[i][j] = func(data[i][j]);
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    return add(other);
}

Matrix Matrix::operator-(const Matrix& other) const {
    return subtract(other);
}

Matrix Matrix::operator*(const Matrix& other) const {
    return multiply(other, true); // Use element-wise multiplication
}

Matrix Matrix::operator*(double scalar) const {
    return multiply(scalar);
}

Matrix Matrix::operator/(double scalar) const {
    return multiply(1/scalar);
}

double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range.");
    }
    return data[row][col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix indices out of range.");
    }
    return data[row][col];
}

bool Matrix::operator==(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        return false;
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (data[i][j] != other.data[i][j]) {
                return false;
            }
        }
    }
    return true;
}

std::partial_ordering Matrix::operator<=>(const Matrix& other) const {
    size_t totalElements1 = rows * cols;
    size_t totalElements2 = other.rows * other.cols;

    if (totalElements1 != totalElements2) {
        return totalElements1 <=> totalElements2;
    }

    double sum1 = 0;
    double sum2 = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            sum1 += data[i][j];
        }
    }
    for (size_t i = 0; i < other.rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            sum2 += other.data[i][j];
        }
    }

    return sum1 <=> sum2;
}

bool Matrix::isEqual(const Matrix& other, double tolerance) const {
    if (rows != other.rows || cols != other.cols) {
        return false;
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (std::fabs(data[i][j] - other.data[i][j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

void Matrix::randomize(double min, double max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            os << std::setw(8) << matrix.data[i][j] << " ";
        }
        os << "\n";
    }
    return os;
}

std::istream& operator>>(std::istream& is, Matrix& matrix) {
    if (is.tellg() == std::streampos(-1)) {
        // Console input
        std::cout << "Enter the values for a " << matrix.rows << "x" << matrix.cols << " matrix:\n";
        for (size_t i = 0; i < matrix.rows; ++i) {
            for (size_t j = 0; j < matrix.cols; ++j) {
                std::cout << "m[" << i << "][" << j << "] = ";
                is >> matrix.data[i][j];
            }
        }
    } else {
        // File input
        std::vector<std::vector<double>> tempData;
        size_t cols = 0;
        std::string line;

        while (std::getline(is, line)) {
            std::istringstream lineStream(line);
            std::vector<double> row;
            double value;
            while (lineStream >> value) {
                row.push_back(value);
            }
            if (cols == 0) {
                cols = row.size();
            } else if (row.size() != cols) {
                throw std::runtime_error("Inconsistent number of columns in matrix data.");
            }
            tempData.push_back(row);
        }

        matrix.rows = tempData.size();
        matrix.cols = cols;
        matrix.data = tempData;
    }
    return is;
}
