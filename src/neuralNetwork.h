/*
 * nerualNetwork.h
 *
 *  Created on: Sep 17, 2020
 *      Author: wade4
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

using namespace std;

#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include "matrix.h"

typedef double NetworkType;
typedef NetworkType *NetworkTypePtr;
typedef NetworkTypePtr *NetworkTypePtrPtr;

double fRand(double fMin, double fMax);

template<class T>
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

        Matrix<T> *trainingInput;
        Matrix<T> *trainingOutput;

        NetworkType Error;
        NetworkType alpha;
        NetworkType eta;
    public:
        NeuralNetwork(int input, int middle, int output, int example, int bias) {
            trainingInput = NULL;
            trainingOutput = NULL;
            Error = 0.0;
            eta = 0.5;
            alpha = 0.9;
            numTrainingSets = example;
            inputSize = input;
            middleSize = middle;
            outputSize = output;
            biasSize = bias;
            trainBias = numTrainingSets + biasSize;
            inputBias = inputSize + biasSize;
            middleBias = middleSize + biasSize;
            outputBias = outputSize + biasSize;

            trainingSetOrder = new int[numTrainingSets];

            allocateMatrix(&inputMiddleWeights, inputBias, middleBias, fRand(-1.0, 1.0));
            allocateMatrix(&middleLayer, trainBias, middleBias, 0.0);
            allocateMatrix(&middleOutputWeights, middleBias, outputBias, fRand(-1.0, 1.0));
            allocateMatrix(&outputLayer, trainBias, outputBias, 0.0);
            allocateMatrix(&outputLayerDelta, outputBias, 0.0);
            allocateMatrix(&middleLayerDelta, middleBias, 0.0);
            allocateMatrix(&deltaWeightInputMiddle, inputBias, middleBias, 0.0);
            allocateMatrix(&deltaWeightMiddleOutput, middleBias, outputBias, 0.0);
        }

        virtual ~NeuralNetwork() {
            delete[] trainingSetOrder;

            deallocate(&inputMiddleWeights, inputBias);
            deallocate(&middleLayer, trainBias);
            deallocate(&middleOutputWeights, middleBias);
            deallocate(&outputLayer, trainBias);
            delete[] outputLayerDelta;
            delete[] middleLayerDelta;
            deallocate(&deltaWeightMiddleOutput, middleBias);
        }

        void setTrainingData(Matrix<T> *in, Matrix<T> *out) {
            trainingInput = new Matrix<T>(in->getRows() + biasSize, in->getColumns() + biasSize, 0);
            for (auto i = biasSize; i < trainingInput->getRows(); i++) {
                for (auto j = biasSize; j < trainingInput->getColumns(); j++) {
                    (*trainingInput)[i][j] = (*in)[i - biasSize][j - biasSize];
                }
            }

            trainingOutput = new Matrix<T>(out->getRows() + biasSize, out->getColumns() + biasSize, 0);
            for (auto i = biasSize; i < trainingOutput->getRows(); i++) {
                for (auto j = biasSize; j < trainingOutput->getColumns(); j++) {
                    (*trainingOutput)[i][j] = (*out)[i - biasSize][j - biasSize];
                }
            }
        }

        double sigmoid(double x) {
            return 1 / (1 + exp(-x));
        }

        double sigmoidDerivative(double x) {
            return x * (1 - x);
        }

        void shuffle(int *array, int n) {
            if (n > 1) {
                for (auto i = 0; i < n - 1; i++) {
                    size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
                    int t = array[j];
                    array[j] = array[i];
                    array[i] = t;
                }
            }
        }

        void randomizeInput() {
            for (auto p = 0; p < numTrainingSets; p++) {
                trainingSetOrder[p] = p;
            }
            shuffle(trainingSetOrder, numTrainingSets);
        }

        void forwardInputHidden(int p) {
            for (auto j = biasSize; j < middleSize + biasSize; j++) {
                NetworkType activate = inputMiddleWeights[0][j];
                for (auto i = biasSize; i < inputSize + biasSize; i++) {
                    activate += (*trainingInput)[p + 1][i] * inputMiddleWeights[i][j];
                }
                middleLayer[p][j] = sigmoid(activate);
            }
        }

        void forwardHiddenOutput(int p) {
            for (auto k = biasSize; k < outputSize + biasSize; k++) {
                NetworkType activate = middleOutputWeights[0][k];
                for (auto j = biasSize; j < middleSize + biasSize; j++) {
                    activate += middleLayer[p][j] * middleOutputWeights[j][k];
                }
                outputLayer[p][k] = sigmoid(activate); /* Sigmoidal Outputs */
            }
        }

        void computeOutputError(int p) {
            for (auto k = biasSize; k < outputSize + biasSize; k++) {
                NetworkType diff = (*trainingOutput)[p][k] - outputLayer[p][k];
                Error += 0.5 * diff * diff; /* SSE */
                outputLayerDelta[k] = diff * sigmoidDerivative(outputLayer[p][k]); /* Sigmoidal Outputs, SSE */
            }
        }

        void computeHiddenError(int p) {
            for (auto j = biasSize; j < middleSize + biasSize; j++) {
                NetworkType activate = 0.0;
                for (auto k = 1; k <= outputSize; k++) {
                    activate += middleOutputWeights[j][k] * outputLayerDelta[k];
                }
                middleLayerDelta[j] = activate * sigmoidDerivative(middleLayer[p][j]); //hidden[p][j] * (1.0 - hidden[p][j]) ;
            }
        }

        void backpropagateOutput(int p) {
            for (auto k = biasSize; k < outputSize + biasSize; k++) {
                deltaWeightMiddleOutput[0][k] = eta * outputLayerDelta[k] + alpha * deltaWeightMiddleOutput[0][k];
                middleOutputWeights[0][k] += deltaWeightMiddleOutput[0][k];
                for (auto j = biasSize; j < middleSize + biasSize; j++) {
                    deltaWeightMiddleOutput[j][k] = eta * middleLayer[p][j] * outputLayerDelta[k] + alpha * deltaWeightMiddleOutput[j][k];
                    middleOutputWeights[j][k] += deltaWeightMiddleOutput[j][k];
                }
            }
        }

        void backpropagateHidden(int p) {
            for (auto j = biasSize; j < middleSize + biasSize; j++) {
                deltaWeightInputMiddle[0][j] = eta * middleLayerDelta[j] + alpha * deltaWeightInputMiddle[0][j];
                inputMiddleWeights[0][j] += deltaWeightInputMiddle[0][j];
                for (auto i = biasSize; i < inputSize + biasSize; i++) {
                    deltaWeightInputMiddle[i][j] = eta * (*trainingInput)[p + 1][i] * middleLayerDelta[j] + alpha * deltaWeightInputMiddle[i][j];
                    inputMiddleWeights[i][j] += deltaWeightInputMiddle[i][j];
                }
            }
        }
        void printResults(int epoch) {
            printf("\n\nNETWORK DATA - EPOCH %d Error = %f\n\nPat\t", epoch, Error); /* print network outputs */
            for (auto i = biasSize; i < inputSize + biasSize; i++) {
                printf("Input%-4d\t", i - biasSize + 1);
            }
            for (auto k = biasSize; k < outputSize + biasSize; k++) {
                printf("Target%-4d\tOutput%-4d\t", k - biasSize + 1, k - biasSize + 1);
            }
            for (auto p = 0; p < numTrainingSets; p++) {
                printf("\n%d\t", p);
                for (auto i = biasSize; i < inputSize + biasSize; i++) {
                    printf("%f\t", (*trainingInput)[p + 1][i]);
                }
                for (auto k = biasSize; k < outputSize + biasSize; k++) {
                    printf("%f\t%f\t", (*trainingOutput)[p][k], outputLayer[p][k]);
                }
            }
        }

        int train(int numberOfEpochs) {
            int epoch;
            for (epoch = 0; epoch < numberOfEpochs; epoch++) { /* iterate weight updates */
                randomizeInput();
                Error = 0.0;
                for (int np = 0; np < numTrainingSets; np++) { /* repeat for all the training patterns */
                    int p = trainingSetOrder[np];

                    forwardInputHidden(p);
                    forwardHiddenOutput(p);

                    computeOutputError(p);
                    computeHiddenError(p);

                    backpropagateOutput(p);
                    backpropagateHidden(p);
                }
                if (epoch % 100 == 0) {
                    printf("\nEpoch %-5d :   Error = %f", epoch, Error);
                }
                if (Error < 0.0004)
                    break; /* stop learning when 'near enough' */
            }
            return epoch;
        }

        void allocateMatrix(NetworkTypePtrPtr *matrix, int size1, int size2, double value) {
            (*matrix) = new NetworkTypePtr[size1];
            for (auto i = 0; i < size1; i++) {
                (*matrix)[i] = new NetworkType[size2];
                for (auto j = 0; j < size2; j++) {
                    (*matrix)[i][j] = value;
                }
            }
        }

        void allocateMatrix(NetworkTypePtr *matrix, int size1, double value) {
            (*matrix) = new NetworkType[size1];
            for (auto j = 0; j < size1; j++) {
                (*matrix)[j] = value;
            }
        }

        void deallocate(NetworkTypePtrPtr *matrix, int size) {
            for (auto i = 0; i < size; i++) {
                delete[] (*matrix)[i];
            }
            delete[] (*matrix);
            (*matrix) = NULL;
        }
};

#endif /* NEURALNETWORK_H_ */
