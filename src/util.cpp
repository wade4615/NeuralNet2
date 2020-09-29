//============================================================================
// Name        : util.cpp
// Author      : Christopher D. Wade
// Version     : 1.0
// Copyright   : (c) 2020 Christopher D. Wade
// Description : Multihidden layer backpropagation net
//============================================================================
#include"nn.h"

NetworkType& Array::operator[](int index) {
    return elements[index];
}

Matrix::Matrix() {
    elements = NULL;
    rows = 0;
    cols = 0;
    size = 0;
}

Matrix::Matrix(initializer_list<initializer_list<NetworkType>> list) {
    size = 0;
    rows = (int) list.size();
    cols = (int) (list.begin())->size();
    elements = new NetworkTypePtr[rows];
    for (auto i = 0; i < rows; i++) {
        elements[i] = new NetworkType[cols];
        for (auto j = 0; j < cols; j++) {
            elements[i][j] = ((list.begin() + i)->begin())[j];
        }
    }
}

NetworkTypePtr& Matrix::operator[](int index) {
    return elements[index];
}
