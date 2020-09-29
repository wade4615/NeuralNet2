//============================================================================
// Name        : nn.h
// Author      : Christopher D. Wade
// Version     : 1.0
// Copyright   : (c) 2020 Christopher D. Wade
// Description : Multihidden layer backpropagation net
//============================================================================
#ifndef NN_H_
#define NN_H_

//#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
using namespace std;

#define printf __mingw_printf

typedef long double NetworkType;
typedef NetworkType *NetworkTypePtr;
typedef NetworkTypePtr *NetworkTypePtrPtr;

class Array {
    public:
        NetworkTypePtr elements;
        int number;
        long size;
        NetworkType& operator[](int index);
};
typedef Array *ArrayPtr;
typedef Array **ArrayPtrPtr;

class Matrix {
    public:
        NetworkTypePtrPtr elements;
        int rows;
        int cols;
        long size;
        Matrix();
        Matrix(initializer_list<initializer_list<NetworkType>> list);
        NetworkTypePtr& operator[](int index);
};
typedef Matrix *MatrixPtr;
typedef Matrix **MatrixPtrPtr;

typedef struct Settings {
        Array configuration;
        int NumPattern;
        int numberOfLayers;
        int numberOfWeights;
        int *trainingSetOrder;
        int outputLayerIndex;
        Matrix trainingInput;
        Matrix trainingOutput;
} Settings, *SettingsPtr, **SettingsPtrPtr;

class NeuralNetwork {
        int epoch;
        long memory;
        long double Error;
        NetworkType eta;
        NetworkType alpha;
        chrono::steady_clock::time_point tbegin;
        chrono::steady_clock::time_point tend;
        MatrixPtr Layers;
        MatrixPtr Weights;
        Matrix Expected;
        ArrayPtr Delta;
        MatrixPtr DeltaWeight;
        MatrixPtr outputLayer;
        SettingsPtr settings;
    public:
        NeuralNetwork(const int numPatterns, MatrixPtr in, MatrixPtr out, int config[], int size);
        virtual ~NeuralNetwork();
        void train();
        void output();
    protected:
        double sigmoid(double x);
        double sigmoidDerivative(double x);
        void forward(int p);
        void computeError(int p);
        void backPropagate(int p);
    private:
        void randomizeInput();
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
};
#endif /* NN_H_ */
