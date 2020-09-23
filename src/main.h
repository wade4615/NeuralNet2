/*
 * main.h
 *
 *  Created on: Sep 21, 2020
 *      Author: wade4
 */

#ifndef MAIN_H_
#define MAIN_H_

#include "matrix.h"

typedef double NetworkType;
typedef NetworkType *NetworkTypePtr;
typedef NetworkTypePtr *NetworkTypePtrPtr;

typedef struct Array {
        NetworkTypePtr elements;
        int number;
} Array, *ArrayPtr, **ArrayPtrPtr;

typedef struct Matrix {
        NetworkTypePtrPtr elements;
        int rows;
        int cols;
} Matrix, *MatrixPtr, **MatrixPtrPtr;

typedef struct NeuralNetwork {
        MatrixPtr layer;
        MatrixPtr weights;
        int numberOfLayers;
        int numberOfWeights;
        int numberOfMiddleLayers;
        int *configuration;
        Array outputLayerDelta;
        ArrayPtr middleLayerDelta;
} NeuralNetwork, *NeuralNetworkPtr, **NeuralNetworkPtrPtr;

double fRand(double fMin, double fMax);

void test2();

void initialize(int configuration[], int example, int bias);
void shutDown();
void setTrainingData(NeuralMatrix<double> *in, NeuralMatrix<double> *out);
double sigmoid(double x);
double sigmoidDerivative(double x);
void shuffle(int *array, int n);
void randomizeInput();
void forward(int p);
void computeError(int p);
void backpropagate(int p);
void printResults(int epoch);
int train(int numberOfEpochs);
void allocateMatrix(NetworkTypePtrPtr *matrix, int size1, int size2, double value);
void allocateMatrix(ArrayPtr matrix, int size1, double value);
void allocateMatrix(MatrixPtr matrix, int size1, int size2, double value);
void deallocate(NetworkTypePtrPtr *matrix, int size);
void deallocate(MatrixPtr matrix);
void deallocate(ArrayPtr matrix);
#endif /* MAIN_H_ */
