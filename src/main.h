/*
 * main.h
 *
 *  Created on: Sep 21, 2020
 *      Author: wade4
 */

#ifndef MAIN_H_
#define MAIN_H_

typedef double NetworkType;
typedef NetworkType *NetworkTypePtr;
typedef NetworkTypePtr *NetworkTypePtrPtr;

double fRand(double fMin, double fMax);

typedef struct NeuralMatrix {
        NetworkTypePtrPtr elements;
        int rows;
        int cols;
} NeuralMatrix, *NeuralMatrixPtr;

typedef struct Network {
        NeuralMatrixPtr layers;
        NeuralMatrixPtr weights;
        NetworkTypePtr outputLayerDelta;
        NetworkTypePtrPtr in;
        NetworkTypePtrPtr out;
        int *trainingSetOrder;
        int numberOfLayers;
        int numberOfWeightMatrices;
} Network;

int* randomizeInput(int numTrainingSets);
void print(const char *text, Network *network);
void shuffle(int *array, int n);
void allocateMatrix(NeuralMatrix *matrix, int size1, int size2, double value);
void allocateMatrix(NetworkTypePtrPtr *matrix, int size1, int size2);
void setupArchitecture(int numTrainingSets, int configuration[], Network *network);
void train(int configuration[], Network *network, NetworkTypePtrPtr input, NetworkTypePtrPtr output);
void test2();

#endif /* MAIN_H_ */
