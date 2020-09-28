/*
 * nn.h
 *
 *  Created on: Sep 28, 2020
 *      Author: wade4
 */

#ifndef NN_H_
#define NN_H_

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>

typedef double NetworkType;
typedef NetworkType *NetworkTypePtr;
typedef NetworkTypePtr *NetworkTypePtrPtr;

typedef struct Array {
        NetworkTypePtr elements;
        int number;
        long size;
} Array, *ArrayPtr, **ArrayPtrPtr;

typedef struct Matrix {
        NetworkTypePtrPtr elements;
        int rows;
        int cols;
        long size;
} Matrix, *MatrixPtr, **MatrixPtrPtr;

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
