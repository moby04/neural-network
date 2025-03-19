#include <iostream>
#include <fstream>
#include "../include/Matrix.h"

int main() {
    Matrix m1 = Matrix::createIdentityMatrix(3, "IdentityMatrix");
    m1.print();

    Matrix m2(2, 2, "CustomMatrix");
    std::cin >> m2;
    m2.print();

    Matrix m3(3, 3, "RandomMatrix");
    m3.randomize(0.0, 10.0);
    m3.print();

    // Example of using file streams
    std::ofstream outFile("matrix_output.txt");
    if (outFile.is_open()) {
        outFile << m2;
        outFile.close();
    }

    std::cout << "Reading from file...\n";
    std::ifstream inFile("matrix_output.txt");
    if (inFile.is_open()) {
        Matrix m4(2, 2, "FileMatrix");
        inFile >> m4;
        m4.print();
        inFile.close();
    }

    return 0;
}
