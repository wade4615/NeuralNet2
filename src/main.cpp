/*
 ============================================================================
 Name        : nn.cpp
 Author      : Christopher D. Wade
 Version     : 1.0
 Copyright   : (c) 2020 Christopher D. Wade
 Description : A basic implementation of a back propagation neural network
 ============================================================================
 */
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
#include "main.h"

NeuralNetwork network;
int numberOfLayers;

Matrix inputLayer;
Matrix inputMiddleWeights;
Matrix middleLayer;
Matrix middleOutputWeights;
Matrix outputLayer;

Matrix deltaWeightInputMiddle;
Matrix deltaWeightMiddleOutput;
Array outputLayerDelta;
Array middleLayerDelta;

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

NeuralMatrix<double> *trainingInput;
NeuralMatrix<double> *trainingOutput;

NetworkType Error1, Error2;
NetworkType alpha;
NetworkType eta;

void initialize(int configuration[], int example, int bias) {
    trainingInput = NULL;
    trainingOutput = NULL;
    Error1 = 0.0;
    eta = 0.5;
    alpha = 0.9;
    numTrainingSets = example;
    inputSize = configuration[0];
    middleSize = configuration[1];
    outputSize = configuration[2];
    biasSize = bias;
    trainBias = numTrainingSets + biasSize;
    inputBias = inputSize + biasSize;
    middleBias = middleSize + biasSize;
    outputBias = outputSize + biasSize;

    trainingSetOrder = new int[numTrainingSets];

    network.configuration = new int[numberOfLayers];
    network.numberOfLayers = numberOfLayers;
    network.numberOfWeights = numberOfLayers - 1;
    network.numberOfMiddleLayers = numberOfLayers - 2;

    network.layer = new Matrix[network.numberOfLayers];
    network.weights = new Matrix[network.numberOfWeights];
    network.middleLayerDelta = new Array[network.numberOfMiddleLayers];

    for (auto i = 0; i < network.numberOfLayers; i++) {
        network.configuration[i] = configuration[i];
        allocateMatrix(&network.layer[i], trainBias, configuration[i] + biasSize, 0.0);
    }
    for (auto i = 0; i < network.numberOfWeights; i++) {
        allocateMatrix(&network.weights[i], configuration[i] + biasSize, configuration[i + 1] + biasSize, fRand(-1.0, 1.0));
    }
    for (auto i = 1; i < network.numberOfMiddleLayers; i++) {
        allocateMatrix(&network.middleLayerDelta[i], configuration[i] + biasSize, 0.0);
    }
    allocateMatrix(&network.outputLayerDelta, outputBias, 0.0);

    allocateMatrix(&inputLayer, trainBias, inputBias, 0.0);
    allocateMatrix(&inputMiddleWeights, inputBias, middleBias, fRand(-1.0, 1.0));
    allocateMatrix(&middleLayer, trainBias, middleBias, 0.0);
    allocateMatrix(&middleOutputWeights, middleBias, outputBias, fRand(-1.0, 1.0));
    allocateMatrix(&outputLayer, trainBias, outputBias, 0.0);

    allocateMatrix(&deltaWeightInputMiddle, inputBias, middleBias, 0.0);
    allocateMatrix(&deltaWeightMiddleOutput, middleBias, outputBias, 0.0);

    allocateMatrix(&outputLayerDelta, outputBias, 0.0);
    allocateMatrix(&middleLayerDelta, middleBias, 0.0);
}

void shutDown() {
    delete[] trainingSetOrder;
    deallocate(&inputLayer);
    deallocate(&inputMiddleWeights);
    deallocate(&middleLayer);
    deallocate(&middleOutputWeights);
    deallocate(&outputLayer);
    delete[] outputLayerDelta.elements;
    delete[] middleLayerDelta.elements;
    deallocate(&deltaWeightMiddleOutput);
    for (auto i = 0; i < network.numberOfLayers; i++) {
        deallocate(&network.layer[i]);
    }
    for (auto i = 0; i < network.numberOfWeights; i++) {
        deallocate(&network.weights[i]);
    }
    for (auto i = 1; i < network.numberOfMiddleLayers; i++) {
        deallocate(&network.middleLayerDelta[i]);
    }
}

void setTrainingData(NeuralMatrix<double> *in, NeuralMatrix<double> *out) {
    trainingInput = new NeuralMatrix<double>(in->getRows() + biasSize, in->getColumns() + biasSize, 0);
    for (auto i = biasSize; i < trainingInput->getRows(); i++) {
        for (auto j = biasSize; j < trainingInput->getColumns(); j++) {
            (*trainingInput)[i][j] = (*in)[i - biasSize][j - biasSize];
        }
    }

    trainingOutput = new NeuralMatrix<double>(out->getRows() + biasSize, out->getColumns() + biasSize, 0);
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

void forward(int p) {
    for (auto j = biasSize; j < middleSize + biasSize; j++) {
        NetworkType activate = inputMiddleWeights.elements[0][j];
        for (auto i = biasSize; i < inputSize + biasSize; i++) {
            activate += inputLayer.elements[p + 1][i] * inputMiddleWeights.elements[i][j];
        }
        middleLayer.elements[p][j] = sigmoid(activate);
    }
    for (auto k = biasSize; k < outputSize + biasSize; k++) {
        NetworkType activate = middleOutputWeights.elements[0][k];
        for (auto j = biasSize; j < middleSize + biasSize; j++) {
            activate += middleLayer.elements[p][j] * middleOutputWeights.elements[j][k];
        }
        outputLayer.elements[p][k] = sigmoid(activate);
    }
    for (auto i = 0; i < network.numberOfWeights; i++) {
        for (auto j = biasSize; j < network.configuration[i + 1] + biasSize; j++) {
            NetworkType activate = network.weights[i].elements[0][j];
            for (auto k = biasSize; k < network.configuration[i] + biasSize; k++) {
                activate += network.layer[i].elements[p + 1][k] * network.weights[i].elements[k][j];
            }
            network.layer[i + 1].elements[p][j] = sigmoid(activate);
        }
    }
}

void computeError(int p) {
    for (auto k = biasSize; k < outputSize + biasSize; k++) {
        NetworkType diff = (*trainingOutput)[p][k] - outputLayer.elements[p][k];
        Error1 += 0.5 * diff * diff;
        outputLayerDelta.elements[k] = diff * sigmoidDerivative(outputLayer.elements[p][k]);
    }
    for (auto j = biasSize; j < middleSize + biasSize; j++) {
        NetworkType activate = 0.0;
        for (auto k = 1; k <= outputSize; k++) {
            activate += middleOutputWeights.elements[j][k] * outputLayerDelta.elements[k];
        }
        middleLayerDelta.elements[j] = activate * sigmoidDerivative(middleLayer.elements[p][j]);
    }
    for (auto k = biasSize; k < outputSize + biasSize; k++) {
        NetworkType diff = (*trainingOutput)[p][k] - network.layer[network.numberOfLayers - 1].elements[p][k];
        Error2 += 0.5 * diff * diff;
        network.outputLayerDelta.elements[k] = diff * sigmoidDerivative(network.layer[network.numberOfLayers - 1].elements[p][k]);
    }
}

void backpropagate(int p) {
    for (auto k = biasSize; k < outputSize + biasSize; k++) {
        deltaWeightMiddleOutput.elements[0][k] = eta * outputLayerDelta.elements[k] + alpha * deltaWeightMiddleOutput.elements[0][k];
        middleOutputWeights.elements[0][k] += deltaWeightMiddleOutput.elements[0][k];
        for (auto j = biasSize; j < middleSize + biasSize; j++) {
            deltaWeightMiddleOutput.elements[j][k] = eta * middleLayer.elements[p][j] * outputLayerDelta.elements[k] + alpha * deltaWeightMiddleOutput.elements[j][k];
            middleOutputWeights.elements[j][k] += deltaWeightMiddleOutput.elements[j][k];
        }
    }
    for (auto j = biasSize; j < middleSize + biasSize; j++) {
        deltaWeightInputMiddle.elements[0][j] = eta * middleLayerDelta.elements[j] + alpha * deltaWeightInputMiddle.elements[0][j];
        inputMiddleWeights.elements[0][j] += deltaWeightInputMiddle.elements[0][j];
        for (auto i = biasSize; i < inputSize + biasSize; i++) {
            deltaWeightInputMiddle.elements[i][j] = eta * inputLayer.elements[p + 1][i] * middleLayerDelta.elements[j] + alpha * deltaWeightInputMiddle.elements[i][j];
            inputMiddleWeights.elements[i][j] += deltaWeightInputMiddle.elements[i][j];
        }
    }
}
void printResults(int epoch) {
    printf("\n\nNETWORK DATA - EPOCH %d Error = %f\n\nPat\t", epoch, Error1); /* print network outputs */
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
            printf("%f\t%f\t", (*trainingOutput)[p][k], outputLayer.elements[p][k]);
        }
    }
}

int train(int numberOfEpochs) {
    int epoch;
    for (epoch = 0; epoch < numberOfEpochs; epoch++) { /* iterate weight updates */
        randomizeInput();
        Error1 = 0.0;
        for (int np = 0; np < numTrainingSets; np++) { /* repeat for all the training patterns */
            int p = trainingSetOrder[np];
            for (auto j = 0; j < inputBias; j++) {
                inputLayer.elements[p + 1][j] = (*trainingInput)[p + 1][j];
                network.layer[0].elements[p + 1][j] = (*trainingInput)[p + 1][j];
            }
            forward(p);
            computeError(p);
            backpropagate(p);
        }
        if (epoch % 100 == 0) {
            printf("\nEpoch %-5d :   Error = %f", epoch, Error1);
        }
        if (Error1 < 0.0004)
            break;
    }
    return epoch;
}

void allocateMatrix(MatrixPtr matrix, int size1, int size2, double value) {
    matrix->rows = size1;
    matrix->cols = size2;
    allocateMatrix(&matrix->elements, size1, size2, value);
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

void allocateMatrix(ArrayPtr matrix, int size1, double value) {
    matrix->elements = new NetworkType[size1];
    matrix->number = size1;
    for (auto j = 0; j < size1; j++) {
        matrix->elements[j] = value;
    }
}

void deallocate(NetworkTypePtrPtr *matrix, int size) {
    for (auto i = 0; i < size; i++) {
        delete[] (*matrix)[i];
    }
    delete[] (*matrix);
    (*matrix) = NULL;
}

void deallocate(MatrixPtr matrix) {
    deallocate(&matrix->elements, matrix->rows);
}

void deallocate(ArrayPtr matrix) {
    delete[] matrix->elements;
    matrix->elements = NULL;
}

int main() {
    NeuralMatrix<double> trainingInput = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
    NeuralMatrix<double> trainingOutput = { { 0 }, { 1 }, { 1 }, { 0 } };

    int configuration[] = { 2, 2, 1 };
    numberOfLayers = sizeof(configuration) / sizeof(int);
    if (numberOfLayers < 3) {
        printf("at least 3 layer needed, %d supplied\n", numberOfLayers);
        exit(-1);
    }
    initialize(configuration, 4, 1);
    setTrainingData(&trainingInput, &trainingOutput);
    int epoch = train(1000000);
    printResults(epoch);
    shutDown();
    printf("\n\nGoodbye!\n\n");
    return 1;
}
