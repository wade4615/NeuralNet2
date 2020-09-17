/*
 * neuralNetwork.cpp
 *
 *  Created on: Sep 17, 2020
 *      Author: wade4
 */
#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include "nn.h"
#include"matrix.h"
#include "neuralNetwork.h"

NeuralNetwork::NeuralNetwork(int input, int middle, int output, int example) {
    trainingInput = NULL;
    trainingOutput = NULL;
    Error = 0.0;
    eta = 0.5;
    alpha = 0.9;
    numTrainingSets = example;
    inputSize = input;
    middleSize = middle;
    outputSize = output;
    trainingSetOrder = new int[numTrainingSets];
    inputMiddleWeights = new NetworkTypePtr[inputSize + 1];
    for (auto i = 0; i < inputSize + 1; i++) {
        inputMiddleWeights[i] = new NetworkType[middleSize + 1];
    }
    middleLayer = new NetworkTypePtr[numTrainingSets + 1];
    for (auto i = 0; i < numTrainingSets + 1; i++) {
        middleLayer[i] = new NetworkType[middleSize + 1];
    }
    middleOutputWeights = new NetworkTypePtr[middleSize + 1];
    for (auto i = 0; i < middleSize + 1; i++) {
        middleOutputWeights[i] = new NetworkType[outputSize + 1];
    }
    outputLayer = new NetworkTypePtr[numTrainingSets + 1];
    for (auto i = 0; i < numTrainingSets + 1; i++) {
        outputLayer[i] = new NetworkType[outputSize + 1];
    }
    outputLayerDelta = new NetworkType[outputSize + 1];
    middleLayerDelta = new NetworkType[middleSize + 1];
    deltaWeightInputMiddle = new NetworkTypePtr[inputSize + 1];
    for (auto i = 0; i < inputSize + 1; i++) {
        deltaWeightInputMiddle[i] = new NetworkType[middleSize + 1];
    }
    deltaWeightMiddleOutput = new NetworkTypePtr[middleSize + 1];
    for (auto i = 0; i < middleSize + 1; i++) {
        deltaWeightMiddleOutput[i] = new NetworkType[outputSize + 1];
    }
}

NeuralNetwork::~NeuralNetwork() {
    delete[] trainingSetOrder;
    for (auto i = 0; i < inputSize + 1; i++) {
        delete[] inputMiddleWeights[i];
    }
    delete[] inputMiddleWeights;
    for (auto i = 0; i < numTrainingSets + 1; i++) {
        delete[] middleLayer[i];
    }
    delete[] middleLayer;
    for (auto i = 0; i < middleSize + 1; i++) {
        delete[] middleOutputWeights[i];
    }
    delete[] middleOutputWeights;
    for (auto i = 0; i < numTrainingSets + 1; i++) {
        delete[] outputLayer[i];
    }
    delete[] outputLayer;
    delete[] outputLayerDelta;
    delete[] middleLayerDelta;
    for (auto i = 0; i < middleSize + 1; i++) {
        delete[] deltaWeightMiddleOutput[i];
    }
    delete[] deltaWeightMiddleOutput;
}

void NeuralNetwork::setTrainingData(Matrix *in, Matrix *out) {
    trainingInput = in;
    trainingOutput = out;
}

double NeuralNetwork::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x) {
    return x * (1 - x);
}

double fRand(double fMin, double fMax) {
    double f = (double) rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void NeuralNetwork::shuffle(int *array, int n) {
    if (n > 1) {
        for (auto i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

/* initialize WeightInputHidden and DeltaWeightInputHidden */
void NeuralNetwork::initializeInputHidden() {
    for (auto j = 1; j <= middleSize; j++) {
        for (auto i = 0; i <= inputSize; i++) {
            deltaWeightInputMiddle[i][j] = 0.0;
            inputMiddleWeights[i][j] = fRand(-1.0, 1.0);
        }
    }
}

/* initialize hiddenOutputWeights and DeltaWeightHiddenOutput */
void NeuralNetwork::initializeHiddenOutput() {
    for (auto k = 1; k <= outputSize; k++) {
        for (auto j = 0; j <= middleSize; j++) {
            deltaWeightMiddleOutput[j][k] = 0.0;
            middleOutputWeights[j][k] = fRand(-1.0, 1.0);
        }
    }
}

/* randomize order of training patterns */
void NeuralNetwork::randomizeInput() {
    for (auto p = 0; p < numTrainingSets; p++) {
        trainingSetOrder[p] = p;
    }
    shuffle(trainingSetOrder, numTrainingSets);
}

/* compute hidden unit activations */
void NeuralNetwork::forwardInputHidden(int p) {
    for (auto j = 1; j <= middleSize; j++) {
        NetworkType activate = inputMiddleWeights[0][j];
        for (auto i = 1; i <= inputSize; i++) {
            activate += (*trainingInput)[p + 1][i] * inputMiddleWeights[i][j];
        }
        middleLayer[p][j] = sigmoid(activate);
    }
}

/* compute output unit activations */
void NeuralNetwork::forwardHiddenOutput(int p) {
    for (auto k = 1; k <= outputSize; k++) {
        NetworkType activate = middleOutputWeights[0][k];
        for (auto j = 1; j <= middleSize; j++) {
            activate += middleLayer[p][j] * middleOutputWeights[j][k];
        }
        outputLayer[p][k] = sigmoid(activate); /* Sigmoidal Outputs */
    }
}

/* compute output unit errors */
void NeuralNetwork::computeOutputError(int p) {
    for (auto k = 1; k <= outputSize; k++) {
        NetworkType diff = (*trainingOutput)[p][k] - outputLayer[p][k];
        Error += 0.5 * diff * diff; /* SSE */
        outputLayerDelta[k] = diff * sigmoidDerivative(outputLayer[p][k]); /* Sigmoidal Outputs, SSE */
    }
}

/* 'back-propagate' errors to hidden layer */
void NeuralNetwork::computeHiddenError(int p) {
    for (auto j = 1; j <= middleSize; j++) {
        NetworkType activate = 0.0;
        for (auto k = 1; k <= outputSize; k++) {
            activate += middleOutputWeights[j][k] * outputLayerDelta[k];
        }
        middleLayerDelta[j] = activate * sigmoidDerivative(middleLayer[p][j]); //hidden[p][j] * (1.0 - hidden[p][j]) ;
    }
}

/* update weights hiddenOutputWeights */
void NeuralNetwork::backpropagateOutput(int p) {
    for (auto k = 1; k <= outputSize; k++) {
        deltaWeightMiddleOutput[0][k] = eta * outputLayerDelta[k] + alpha * deltaWeightMiddleOutput[0][k];
        middleOutputWeights[0][k] += deltaWeightMiddleOutput[0][k];
        for (auto j = 1; j <= middleSize; j++) {
            deltaWeightMiddleOutput[j][k] = eta * middleLayer[p][j] * outputLayerDelta[k] + alpha * deltaWeightMiddleOutput[j][k];
            middleOutputWeights[j][k] += deltaWeightMiddleOutput[j][k];
        }
    }
}

/* update weights inputHiddenWeights */
void NeuralNetwork::backpropagateHidden(int p) {
    for (auto j = 1; j <= middleSize; j++) {
        deltaWeightInputMiddle[0][j] = eta * middleLayerDelta[j] + alpha * deltaWeightInputMiddle[0][j];
        inputMiddleWeights[0][j] += deltaWeightInputMiddle[0][j];
        for (auto i = 1; i <= inputSize; i++) {
            deltaWeightInputMiddle[i][j] = eta * (*trainingInput)[p + 1][i] * middleLayerDelta[j] + alpha * deltaWeightInputMiddle[i][j];
            inputMiddleWeights[i][j] += deltaWeightInputMiddle[i][j];
        }
    }
}

void NeuralNetwork::printResults(int epoch) {
    printf("\n\nNETWORK DATA - EPOCH %d Error = %f\n\nPat\t", epoch, Error); /* print network outputs */
    for (auto i = 1; i <= inputSize; i++) {
        printf("Input%-4d\t", i);
    }
    for (auto k = 1; k <= outputSize; k++) {
        printf("Target%-4d\tOutput%-4d\t", k, k);
    }
    for (auto p = 0; p < numTrainingSets; p++) {
        printf("\n%d\t", p);
        for (auto i = 1; i <= inputSize; i++) {
            printf("%f\t", (*trainingInput)[p + 1][i]);
        }
        for (auto k = 1; k <= outputSize; k++) {
            printf("%f\t%f\t", (*trainingOutput)[p][k], outputLayer[p][k]);
        }
    }
}

int NeuralNetwork::train(int numberOfEpochs) {
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

