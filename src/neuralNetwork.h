/*
 * nerualNetwork.h
 *
 *  Created on: Sep 17, 2020
 *      Author: wade4
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

using namespace std;

#include "matrix.h"

typedef double NetworkType;
typedef NetworkType *NetworkTypePtr;
typedef NetworkTypePtr *NetworkTypePtrPtr;

double fRand(double fMin, double fMax);

class NeuralNetwork {
        NetworkTypePtrPtr inputMiddleWeights;
        NetworkTypePtrPtr middleLayer;
        NetworkTypePtrPtr middleOutputWeights;
        NetworkTypePtrPtr outputLayer;
        NetworkTypePtrPtr deltaWeightInputMiddle;
        NetworkTypePtrPtr deltaWeightMiddleOutput;

        NetworkTypePtr outputLayerDelta;
        NetworkTypePtr middleLayerDelta;

        int *trainingSetOrder;
        int numTrainingSets;
        int inputSize;
        int middleSize;
        int outputSize;
        int biasSize;

        int trainBias;
        int inputBias;
        int middleBias;
        int outputBias;

        Matrix<double> *trainingInput;
        Matrix<double> *trainingOutput;

        NetworkType Error;
        NetworkType alpha;
        NetworkType eta;
    public:
        NeuralNetwork(int input, int middle, int output, int example, int bias);
        virtual ~NeuralNetwork();
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
        void setTrainingData(Matrix<double> *in, Matrix<double> *out);
        void allocateMatrix(NetworkTypePtrPtr *matrix, int size1, int size2, double value);
        void allocateMatrix(NetworkTypePtr *matrix, int size1, double value);
        void deallocate(NetworkTypePtrPtr *matrix, int size);
};

#endif /* NEURALNETWORK_H_ */
