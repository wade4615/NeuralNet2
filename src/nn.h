/*
 * nn.h
 *
 *  Created on: Sep 17, 2020
 *      Author: wade4
 */

#ifndef NN_H_
#define NN_H_

#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>

#define NUMBER_OF_EXAMPLES 4
#define INPUT_LAYER_SIZE  2
#define HIDDEN_LAYER_SIZE 2
#define OUTPUT_LAYER_SIZE 1

typedef double NetworkType;
typedef NetworkType *NetworkTypePtr;
typedef NetworkTypePtr *NetworkTypePtrPtr;

using namespace std;

class NeuralNetwork {
        NetworkTypePtrPtr inputMiddleWeights;
        NetworkTypePtrPtr middleLayer;
        NetworkTypePtrPtr middleOutputWeights;
        NetworkTypePtrPtr outputLayer;
        NetworkTypePtr outputLayerDelta;
        NetworkTypePtr middleLayerDelta;
        NetworkTypePtrPtr deltaWeightInputMiddle;
        NetworkTypePtrPtr deltaWeightMiddleOutput;
        int *trainingSetOrder;
        int numTrainingSets;
        int inputSize;
        int middleSize;
        int outputSize;
        NetworkTypePtrPtr trainingInput;
        NetworkTypePtrPtr trainingOutput;
        NetworkType Error;
        NetworkType alpha;
        NetworkType eta;
    public:
        NeuralNetwork(int input, int middle, int output, int example);
        virtual ~NeuralNetwork();
        double sigmoid(double x);
        double sigmoidDerivative(double x);
        double fRand(double fMin, double fMax);
        void shuffle(int *array, int n);
        void initializeInputHidden();
        void initializeHiddenOutput();
        void randomizeInput();
        void forwardInputHidden(int p);
        void forwardHiddenOutput(int p);
        void computeOutputError(int p);
        void computeHiddenError(int p);
        void backpropagateOutput(int p);
        void backpropagateHidden(int p);
        void printResults(int epoch);
        int train(int numberOfEpochs);
        void setTrainingData(initializer_list<initializer_list<double>> in, initializer_list<initializer_list<double>> out);
};
#endif /* NN_H_ */
