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
#include "matrix.h"
#include "neuralNetwork.h"

using namespace std;

typedef struct NeuralMatrix {
        double **elements;
        int rows;
        int cols;
} NeuralMatrix;

typedef struct Network {
        NeuralMatrix *layers;
        NeuralMatrix *weights;
        int numberOfLayers;
        int numberOfWeightMatrices;
} Network;

void test1() {
    Matrix<double> trainingInput = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
    Matrix<double> trainingOutput = { { 0 }, { 1 }, { 1 }, { 0 } };
    NeuralNetwork<double> network(2, 4, 1, 4, 1);
    network.setTrainingData(&trainingInput, &trainingOutput);
    int epoch = network.train(1000000);
    network.printResults(epoch);
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
        printf("\n\n");
    }
    printf("\n");
    for (auto i = 0; i < network->numberOfWeightMatrices; i++) {
        printf("%s for weight %d\n", text, i);
        for (auto k = 0; k < network->weights[i].rows; k++) {
            for (auto l = 0; l < network->weights[i].cols; l++) {
                printf("%1.4f ", network->weights[i].elements[k][l]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
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

void setupArchitecture(int numTrainingSets, int configuration[], Network *network) {
    printf("network.Layers=%d\n", network->numberOfLayers);
    printf("network.numberOfWeightMatrices=%d\n", network->numberOfWeightMatrices);

    network->layers = new NeuralMatrix[network->numberOfLayers];
    network->weights = new NeuralMatrix[network->numberOfWeightMatrices];

    int configIndex = 0;
    for (auto i = 0; i < network->numberOfLayers; i++) {
        int numberOfNeurons = configuration[configIndex];
        allocateMatrix(&network->layers[i], numTrainingSets, numberOfNeurons, 0.0);
        printf("layer %d rows %d cols %d\n", i, network->layers[i].rows, network->layers[i].cols);
        configIndex++;
    }
    for (auto i = 0; i < network->numberOfWeightMatrices; i++) {
        allocateMatrix(&network->weights[i], network->layers[i].cols, network->layers[i + 1].cols, 1.0);
        printf("weights %d rows %d cols %d\n", i, network->weights[i].rows, network->weights[i].cols);
    }
}

void test2() {
    int configuration[] = { 2, 3, 4, 1 };
    int numTrainingSets = 4;
    //double input[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    //double output[4][1] = { { 0 }, { 1 }, { 1 }, { 0 } };
    int *trainingSetOrder = new int[numTrainingSets];

    for (auto p = 0; p < numTrainingSets; p++) {
        trainingSetOrder[p] = p;
    }
    shuffle(trainingSetOrder, numTrainingSets);

    Network network;
    network.numberOfLayers = sizeof(configuration) / sizeof(int);
    network.numberOfWeightMatrices = network.numberOfLayers - 1;
    setupArchitecture(numTrainingSets, configuration, &network);
    printf("\n\n");
    print("before", &network);
//    for (auto i = 0; i < network.numberOfWeightMatrices; i++) {
//        for (auto j = 0; j < network.layers[i + 1].number; j++) {
//            NetworkType activate = 0;
//            for (auto k = 0; k < network.layers[i].number; k++) {
//                activate += network.layers[i].neurons[k] * network.weights[i].elements[k][j];
//            }
//            network.layers[i + 1].neurons[j] = activate;
//        }
//    }
//    print("after", &network);
}

int main() {
    test2();
    printf("\n\nGoodbye!\n\n");
    return 1;
}
