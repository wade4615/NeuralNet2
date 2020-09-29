//============================================================================
// Name        : nn.h
// Author      : Christopher D. Wade
// Version     : 1.0
// Copyright   : (c) 2020 Christopher D. Wade
// Description : Multihidden layer backpropagation net
//============================================================================
#ifndef NN_H_
#define NN_H_

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#define printf __mingw_printf

typedef long double NetworkType;
typedef NetworkType *NetworkTypePtr;
typedef NetworkTypePtr *NetworkTypePtrPtr;

typedef struct Array {
        NetworkTypePtr elements;
        int number;
        long size;
        NetworkType& operator[](int index);
} Array, *ArrayPtr, **ArrayPtrPtr;

typedef struct Matrix {
        NetworkTypePtrPtr elements;
        int rows;
        int cols;
        long size;
} Matrix, *MatrixPtr, **MatrixPtrPtr;

typedef struct Settings {
        Array configuration;
        int NumPattern;
        int numberOfLayers;
        int numberOfWeights;
} Settings, *SettingsPtr, **SettingsPtrPtr;

NetworkType fRand(NetworkType fMin, NetworkType fMax);

long allocateMatrix(NetworkTypePtrPtr *matrix, int size1, int size2, double low, double high);
long allocateMatrix(NetworkTypePtrPtr *matrix, int size1, int size2, double value);
void allocateMatrix(MatrixPtr matrix, int size1, int size2, double low, double high);
void allocateMatrix(MatrixPtr matrix, int size1, int size2, double value);
void allocateMatrix(ArrayPtr matrix, int size1, double value);
long allocateMatrix(NetworkTypePtr *matrix, int size1, double value);
void deallocate(NetworkTypePtrPtr *matrix, int size);
void deallocate(NetworkTypePtr *matrix);
void deallocate(MatrixPtr matrix);
void deallocate(ArrayPtr matrix);

#endif /* NN_H_ */
