/*
 * Matrix.cpp
 *
 *  Created on: Sep 17, 2020
 *      Author: wade4
 */
#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include "matrix.h"

Matrix::Matrix(int rowSize, int colSize, double low, double high) {
    m_rowSize = rowSize;
    m_colSize = colSize;
    elements = new NetworkTypePtr[rowSize];
    for (auto i = 0; i < rowSize; i++) {
        elements[i] = new NetworkType[colSize];
        for (auto j = 0; j < m_colSize; j++) {
            elements[i][j] = fRand(low, high);
        }
    }
}

Matrix::Matrix(int rowSize, int colSize, double value) {
    m_rowSize = rowSize;
    m_colSize = colSize;
    elements = new NetworkTypePtr[rowSize];
    for (auto i = 0; i < rowSize; i++) {
        elements[i] = new NetworkType[colSize];
        for (auto j = 0; j < m_colSize; j++) {
            elements[i][j] = value;
        }
    }
}

Matrix::Matrix(initializer_list<initializer_list<double>> list) {
    m_rowSize = (int) list.size();
    m_colSize = (int) (list.begin())->size();
    elements = new NetworkTypePtr[m_rowSize];
    for (auto i = 0; i < m_rowSize; i++) {
        elements[i] = new NetworkType[m_colSize];
        for (auto j = 0; j < m_colSize; j++) {
            elements[i][j] = ((list.begin() + i)->begin())[j];
        }
    }
}

Matrix::~Matrix() {
    for (auto i = 0; i < m_rowSize + 1; i++) {
        delete[] elements[i];
    }
    delete[] elements;
}

NetworkTypePtr& Matrix::operator[](const int &index) {
    return elements[index];
}

int Matrix::getRows() const {
    return m_rowSize;
}

int Matrix::getCols() const {
    return m_colSize;
}

