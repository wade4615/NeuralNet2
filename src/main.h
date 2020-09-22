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

double fRand(double fMin, double fMax);

void test2();

void initialize(int input, int middle, int output, int example, int bias);
void shutDown();
void setTrainingData(NeuralMatrix<double> *in, NeuralMatrix<double> *out);
double sigmoid(double x);
double sigmoidDerivative(double x);
void shuffle(int *array, int n);
void randomizeInput();
void forwardInputHidden(int p);
void forwardHiddenOutput(int p);
void computeOutputError(int p);
void computeHiddenError(int p);
void backpropagateOutput(int p);
void backpropagateHidden(int p);
void printResults(int epoch);
int train(int numberOfEpochs);
void allocateMatrix(NetworkTypePtrPtr *matrix, int size1, int size2, double value);
void allocateMatrix(NetworkTypePtr *matrix, int size1, double value);
void deallocate(NetworkTypePtrPtr *matrix, int size);

#endif /* MAIN_H_ */
