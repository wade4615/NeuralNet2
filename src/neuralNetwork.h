/*
 * nerualNetwork.h
 *
 *  Created on: Sep 17, 2020
 *      Author: wade4
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "nn.h"
#include "matrix.h"

using namespace std;

double fRand(double fMin, double fMax);

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
        Matrix *trainingInput;
        Matrix *trainingOutput;
        NetworkType Error;
        NetworkType alpha;
        NetworkType eta;
    public:
        NeuralNetwork(int input, int middle, int output, int example);
        virtual ~NeuralNetwork();
        double sigmoid(double x);
        double sigmoidDerivative(double x);
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
        void setTrainingData(Matrix *in, Matrix *out);
};

#endif /* NEURALNETWORK_H_ */
