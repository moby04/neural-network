#include <gtest/gtest.h>
#include "../include/Matrix.h"
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>

TEST(MatrixTest, CreateIdentityMatrix) {
    Matrix m = Matrix::createIdentityMatrix(3, "IdentityMatrix");
    std::ostringstream oss;
    oss << m;
    std::string expected_output = "       1        0        0 \n"
                                  "       0        1        0 \n"
                                  "       0        0        1 \n";
    EXPECT_EQ(oss.str(), expected_output);
}

TEST(MatrixTest, SetName) {
    Matrix m(2, 2, "InitialName");
    m.setName("NewName");
    EXPECT_EQ(m.getName(), "NewName");
}

TEST(MatrixTest, FillByHand) {
    Matrix m(2, 2, "TestMatrix");
    std::istringstream iss("1 2\n3 4");
    iss >> m;
    std::ostringstream oss;
    oss << m;
    std::string expected_output = "       1        2 \n"
                                  "       3        4 \n";
    EXPECT_EQ(oss.str(), expected_output);
}

TEST(MatrixTest, Print) {
    Matrix m(2, 2, "PrintMatrix");
    std::istringstream iss("5 6\n7 8");
    iss >> m;
    std::ostringstream oss;
    std::streambuf* oldCoutStreamBuf = std::cout.rdbuf();
    std::cout.rdbuf(oss.rdbuf());
    m.print();
    std::cout.rdbuf(oldCoutStreamBuf);
    std::string expected_output = "Matrix name: PrintMatrix\n"
                                  "       5        6 \n"
                                  "       7        8 \n";
    EXPECT_EQ(oss.str(), expected_output);
}

TEST(MatrixTest, PrintWithOperator) {
    Matrix m(2, 2, "PrintMatrix");
    std::istringstream iss("5 6\n7 8");
    iss >> m;
    std::ostringstream oss;
    oss << m;
    std::string expected_output = "       5        6 \n"
                                  "       7        8 \n";
    EXPECT_EQ(oss.str(), expected_output);
}

TEST(MatrixTest, FillByHandWithOperator) {
    Matrix m(2, 2, "TestMatrix");
    std::istringstream iss("1 2\n3 4");
    iss >> m;
    std::ostringstream oss;
    oss << m;
    std::string expected_output = "       1        2 \n"
                                  "       3        4 \n";
    EXPECT_EQ(oss.str(), expected_output);
}

TEST(MatrixTest, FileIO) {
    Matrix A(2, 2, "MatrixA");
    std::istringstream iss("1 2\n3 4");
    iss >> A;

    std::ofstream outFile("matrix_test.txt");
    outFile << A;
    outFile.close();

    Matrix B(2, 2, "MatrixB");
    std::ifstream inFile("matrix_test.txt");
    inFile >> B;
    inFile.close();

    EXPECT_TRUE(A.isEqual(B)); // Ensure the loaded matrix matches the saved one
}

TEST(MatrixTest, AddMatrices) {
    Matrix m1(2, 2, "Matrix1");
    std::istringstream iss1("1 2\n3 4");
    iss1 >> m1;

    Matrix m2(2, 2, "Matrix2");
    std::istringstream iss2("5 6\n7 8");
    iss2 >> m2;

    Matrix result = m1 + m2;
    std::ostringstream oss;
    oss << result;
    std::string expected_output = "       6        8 \n"
                                  "      10       12 \n";
    EXPECT_EQ(oss.str(), expected_output);
}

TEST(MatrixTest, SubtractMatrices) {
    Matrix m1(2, 2, "Matrix1");
    std::istringstream iss1("5 6\n7 8");
    iss1 >> m1;

    Matrix m2(2, 2, "Matrix2");
    std::istringstream iss2("1 2\n3 4");
    iss2 >> m2;

    Matrix result = m1 - m2;
    std::ostringstream oss;
    oss << result;
    std::string expected_output = "       4        4 \n"
                                  "       4        4 \n";
    EXPECT_EQ(oss.str(), expected_output);
}

TEST(MatrixTest, TransposeMatrix) {
    Matrix m(2, 3, "Matrix");
    std::istringstream iss("1 2 3\n4 5 6");
    iss >> m;

    Matrix result = m.transpose();
    std::ostringstream oss;
    oss << result;
    std::string expected_output = "       1        4 \n"
                                  "       2        5 \n"
                                  "       3        6 \n";
    EXPECT_EQ(oss.str(), expected_output);
}

TEST(MatrixTest, CompareMatricesEqual) {
    Matrix m1(2, 2, "Matrix1");
    std::istringstream iss1("1 2\n3 4");
    iss1 >> m1;

    Matrix m2(2, 2, "Matrix2");
    std::istringstream iss2("1 2\n3 4");
    iss2 >> m2;

    EXPECT_TRUE(m1 == m2);
}

TEST(MatrixTest, CompareMatricesNotEqual) {
    Matrix m1(2, 2, "Matrix1");
    std::istringstream iss1("1 2\n3 4");
    iss1 >> m1;

    Matrix m2(2, 2, "Matrix2");
    std::istringstream iss2("5 6\n7 8");
    iss2 >> m2;

    EXPECT_FALSE(m1 == m2);
}

TEST(MatrixTest, CompareMatricesThreeWay) {
    Matrix m1(2, 2, "Matrix1");
    std::istringstream iss1("1 2\n3 4");
    iss1 >> m1;

    Matrix m2(2, 2, "Matrix2");
    std::istringstream iss2("1 2\n3 4");
    iss2 >> m2;

    EXPECT_TRUE((m1 <=> m2) == std::partial_ordering::equivalent);

    Matrix m3(2, 2, "Matrix3");
    std::istringstream iss3("0 1\n2 3");
    iss3 >> m3;

    EXPECT_TRUE((m3 <=> m1) == std::partial_ordering::less);

    Matrix m4(2, 2, "Matrix4");
    std::istringstream iss4("2 3\n4 5");
    iss4 >> m4;

    EXPECT_TRUE((m4 <=> m1) == std::partial_ordering::greater);

    Matrix m5(3, 3, "Matrix5");
    std::istringstream iss5("1 2 3\n4 5 6\n7 8 9");
    iss5 >> m5;

    EXPECT_TRUE((m1 <=> m5) == std::partial_ordering::less);
}

TEST(MatrixTest, MultiplyMatrices) {
    Matrix m1(2, 3, "Matrix1");
    std::istringstream iss1("1 2 3\n4 5 6");
    iss1 >> m1;

    Matrix m2(3, 2, "Matrix2");
    std::istringstream iss2("7 8\n9 10\n11 12");
    iss2 >> m2;

    Matrix result = m1.multiply(m2, false); // Regular matrix multiplication
    std::ostringstream oss;
    oss << result;
    std::string expected_output = "      58       64 \n"
                                  "     139      154 \n";
    EXPECT_EQ(oss.str(), expected_output);
}

TEST(MatrixTest, MultiplyMatricesElementWise) {
    Matrix m1(2, 2, "Matrix1");
    std::istringstream iss1("1 2\n3 4");
    iss1 >> m1;

    Matrix m2(2, 2, "Matrix2");
    std::istringstream iss2("5 6\n7 8");
    iss2 >> m2;

    Matrix result = m1 * m2; // Element-wise multiplication
    std::ostringstream oss;
    oss << result;
    std::string expected_output = "       5       12 \n"
                                  "      21       32 \n";
    EXPECT_EQ(oss.str(), expected_output);
}

TEST(MatrixTest, MultiplyMatrixByScalar) {
    Matrix m(2, 2, "Matrix");
    std::istringstream iss("1 2\n3 4");
    iss >> m;

    Matrix result = m * 2.0f;
    std::ostringstream oss;
    oss << result;
    std::string expected_output = "       2        4 \n"
                                  "       6        8 \n";
    EXPECT_EQ(oss.str(), expected_output);
}

TEST(MatrixTest, ApplyFunction) {
    Matrix m(2, 2, "Matrix");
    std::istringstream iss("1 2\n3 4");
    iss >> m;

    auto square = [](float x) { return x * x; };
    m = m.applyFunction(square);

    std::ostringstream oss;
    oss << m;
    std::string expected_output = "       1        4 \n"
                                  "       9       16 \n";
    EXPECT_EQ(oss.str(), expected_output);
}

TEST(MatrixTest, RandomizeMatrix) {
    Matrix m(2, 2, "RandomMatrix");
    m.randomize(0.0f, 1.0f);
    const auto& data = m.getData();
    for (const auto& row : data) {
        for (const auto& value : row) {
            EXPECT_GE(value, 0.0f);
            EXPECT_LE(value, 1.0f);
        }
    }
}

TEST(MatrixTest, ScalarMultiplication) {
    Matrix m(2, 2, "Matrix");
    std::istringstream iss("1 2\n3 4");
    iss >> m;

    Matrix result = m * 2.0f;
    std::ostringstream oss;
    oss << result;
    std::string expected_output = "       2        4 \n"
                                  "       6        8 \n";
    EXPECT_EQ(oss.str(), expected_output);
}
