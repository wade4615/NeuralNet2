/*
 ============================================================================
 Name        : nn.cpp
 Author      : Christopher D. Wade
 Version     : 1.0
 Copyright   : (c) 2020 Christopher D. Wade
 Description : A basic implementation of a back propagation neural network
 ============================================================================
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
#include "main.h"
using namespace std;

NetworkType Error;

int* randomizeInput(int numTrainingSets) {
    int *trainingSetOrder = new int[numTrainingSets];
    for (auto p = 0; p < numTrainingSets; p++) {
        trainingSetOrder[p] = p;
    }
    shuffle(trainingSetOrder, numTrainingSets);
    return trainingSetOrder;
}

void print(const char *text, Network *network) {
    for (auto i = 0; i < network->numberOfLayers; i++) {
        printf("%s for layer %d\n", text, i);
        for (auto k = 0; k < network->layers[i].rows; k++) {
            for (auto l = 0; l < network->layers[i].cols; l++) {
                printf("%1.4f ", network->layers[i].elements[k][l]);
            }
            printf("\n");
        }
    }
//    printf("\n");
//    for (auto i = 0; i < network->numberOfWeightMatrices; i++) {
//        printf("%s for weight %d\n", text, i);
//        for (auto k = 0; k < network->weights[i].rows; k++) {
//            for (auto l = 0; l < network->weights[i].cols; l++) {
//                printf("%1.4f ", network->weights[i].elements[k][l]);
//            }
//            printf("\n");
//        }
//        printf("\n");
//    }
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

void allocateMatrix(NeuralMatrix *matrix, int size1, int size2, double value) {
    matrix->elements = new NetworkTypePtr[size1];
    matrix->rows = size1;
    for (auto i = 0; i < size1; i++) {
        matrix->elements[i] = new NetworkType[size2];
        matrix->cols = size2;
        for (auto j = 0; j < size2; j++) {
            matrix->elements[i][j] = value;
        }
    }
}

void allocateMatrix(NetworkTypePtrPtr *matrix, int size1, int size2) {
    (*matrix) = new NetworkTypePtr[size1];
    for (auto i = 0; i < size1; i++) {
        (*matrix)[i] = new NetworkType[size2];
    }
}

void setupArchitecture(int numTrainingSets, int configuration[], Network *network) {
    printf("network.Layers=%d\n", network->numberOfLayers);
    printf("network.numberOfWeightMatrices=%d\n", network->numberOfWeightMatrices);

    network->layers = new NeuralMatrix[network->numberOfLayers];
    network->weights = new NeuralMatrix[network->numberOfWeightMatrices];

    int configIndex = 0;
    for (auto i = 0; i < network->numberOfLayers; i++) {
        int numberOfNeurons = configuration[configIndex];
        allocateMatrix(&network->layers[i], numTrainingSets, numberOfNeurons, 1.0);
        printf("layer[%d] rows %d cols %d\n", i, network->layers[i].rows, network->layers[i].cols);
        configIndex++;
    }
    for (auto i = 0; i < network->numberOfWeightMatrices; i++) {
        allocateMatrix(&network->weights[i], network->layers[i].cols, network->layers[i + 1].cols, 1.0);
        printf("weights[%d] rows %d cols %d\n", i, network->weights[i].rows, network->weights[i].cols);
    }
    network->outputLayerDelta = new NetworkType[network->numberOfLayers - 1];
    for (auto i = 0; i < network->numberOfLayers; i++) {
        network->outputLayerDelta[i] = 0.0;
        printf("output delta[%d]= %1.4f\n", i, network->outputLayerDelta[i]);
    }
    print("before", network);
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1 - x);
}

void train(int configuration[], Network *network, int numTrainingSets, NetworkTypePtrPtr input, NetworkTypePtrPtr output) {
    for (auto np = 0; np < numTrainingSets; np++) {
        //determine which traing set is going through this time
        int p = network->trainingSetOrder[np];
        //assign  training info to input layer
        for (auto i = 0; i < configuration[0]; i++) {
            network->layers[0].elements[p][i] = input[p][i];
        }
        // feed forward through entire net
        for (auto i = 0; i < network->numberOfWeightMatrices; i++) {
            for (auto j = 0; j < network->layers[i + 1].rows; j++) {
                NetworkType activate = 0;
                for (auto k = 0; k < network->layers[i].cols; k++) {
                    activate += network->layers[i].elements[p][k] * network->weights[i].elements[k][j];
                }
                network->layers[i + 1].elements[p][j] = sigmoid(activate);
            }
        }
        printf("Output Layer Delta for example %d\n", np);
        for (auto k = 0; k < configuration[network->numberOfLayers - 1]; k++) {
            NetworkType diff = output[p][k] - network->layers[network->numberOfLayers - 1].elements[p][k];
            Error += 0.5 * diff * diff; /* SSE */
            network->outputLayerDelta[k] = diff * sigmoidDerivative(network->layers[network->numberOfLayers - 1].elements[p][k]);
            printf("%1.4f ", network->outputLayerDelta[k]);
        }
        printf("\n");
    }
}

void test2() {
    int configuration[] = { 2, 2, 1 };
    int numTrainingSets = 4;
    NetworkType input[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    NetworkType output[4][1] = { { 0 }, { 1 }, { 1 }, { 0 } };

    Network network;
    network.numberOfLayers = sizeof(configuration) / sizeof(int);
    if (network.numberOfLayers < 3) {
        printf("not enough layers in configuration");
        exit(-1);
    }

    NetworkTypePtrPtr in;
    NetworkTypePtrPtr out;
    allocateMatrix(&in, numTrainingSets, configuration[0]);
    allocateMatrix(&out, numTrainingSets, configuration[network.numberOfLayers - 1]);
    for (auto i = 0; i < numTrainingSets; i++) {
        for (auto j = 0; j < configuration[0]; j++) {
            in[i][j] = input[i][j];
        }
    }
    for (auto i = 0; i < numTrainingSets; i++) {
        for (auto j = 0; j < configuration[network.numberOfLayers - 1]; j++) {
            out[i][j] = output[i][j];
        }
    }
    network.trainingSetOrder = randomizeInput(numTrainingSets);

    network.numberOfWeightMatrices = network.numberOfLayers - 1;
    setupArchitecture(numTrainingSets, configuration, &network);
    train(configuration, &network, numTrainingSets, in, out);
    print("after", &network);
}

int main() {
    test2();
    printf("\n\nGoodbye!\n\n");
    return 1;
}
