//============================================================================
// Name        : util.cpp
// Author      : Christopher D. Wade
// Version     : 1.0
// Copyright   : (c) 2020 Christopher D. Wade
// Description : Multihidden layer backpropagation net
//============================================================================
#include"nn.h"

long allocateMatrix(NetworkTypePtrPtr *matrix, int size1, int size2, double low, double high) {
    (*matrix) = new NetworkTypePtr[size1];
    for (auto i = 0; i < size1; i++) {
        (*matrix)[i] = new NetworkType[size2];
        for (auto j = 0; j < size2; j++) {
            (*matrix)[i][j] = fRand(low, high);
        }
    }
    return size1 * size2 * sizeof(NetworkType);
}

long allocateMatrix(NetworkTypePtrPtr *matrix, int size1, int size2, double value) {
    (*matrix) = new NetworkTypePtr[size1];
    for (auto i = 0; i < size1; i++) {
        (*matrix)[i] = new NetworkType[size2];
        for (auto j = 0; j < size2; j++) {
            (*matrix)[i][j] = value;
        }
    }
    return size1 * size2 * sizeof(NetworkType);
}

void allocateMatrix(MatrixPtr matrix, int size1, int size2, double low, double high) {
    matrix->rows = size1;
    matrix->cols = size2;
    matrix->size = allocateMatrix(&matrix->elements, size1, size2, low, high);
}

void allocateMatrix(MatrixPtr matrix, int size1, int size2, double value) {
    matrix->rows = size1;
    matrix->cols = size2;
    matrix->size = allocateMatrix(&matrix->elements, size1, size2, value);
}

void allocateMatrix(ArrayPtr matrix, int size1, double value) {
    matrix->elements = new NetworkType[size1];
    matrix->number = size1;
    for (auto j = 0; j < size1; j++) {
        matrix->elements[j] = value;
    }
    matrix->size = size1 * sizeof(NetworkType);
}

long allocateMatrix(NetworkTypePtr *matrix, int size1, double value) {
    (*matrix) = new NetworkType[size1];
    for (auto j = 0; j < size1; j++) {
        (*matrix)[j] = value;
    }
    return size1 * sizeof(NetworkType);
}

void deallocate(NetworkTypePtrPtr *matrix, int size) {
    for (auto i = 0; i < size; i++) {
        delete[] (*matrix)[i];
    }
    delete[] (*matrix);
    (*matrix) = NULL;
}

void deallocate(NetworkTypePtr *matrix) {
    delete[] (*matrix);
    (*matrix) = NULL;
}

void deallocate(MatrixPtr matrix) {
    deallocate(&matrix->elements, matrix->rows);
}

void deallocate(ArrayPtr matrix) {
    delete[] matrix->elements;
    matrix->elements = NULL;
}
