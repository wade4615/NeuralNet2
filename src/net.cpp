//============================================================================
// Name        : net.cpp
// Author      : Christopher D. Wade
// Version     : 1.0
// Copyright   : (c) 2020 Christopher D. Wade
// Description : Multihidden layer backpropagation net
//============================================================================
#include "nn.h"

NeuralNetwork::NeuralNetwork(const int numPatterns, MatrixPtr in, MatrixPtr out, int config[], int size) {
    eta = 0.5;
    alpha = 0.9;
    settings = new Settings;
    settings->NumPattern = numPatterns;
    settings->numberOfLayers = size;
    settings->numberOfWeights = settings->numberOfLayers - 1;
    settings->trainingSetOrder = new int[settings->NumPattern + 1];
    settings->outputLayerIndex = settings->numberOfLayers - 1;
    allocateMatrix(&settings->configuration, settings->numberOfLayers, 0);
    for (auto i = 0; i < settings->numberOfLayers; i++) {
        settings->configuration[i] = config[i];
    }
    allocateMatrix(&settings->trainingInput, settings->NumPattern, settings->configuration[0], 0);
    allocateMatrix(&settings->trainingOutput, settings->NumPattern, settings->configuration[settings->outputLayerIndex], 0);
    for (auto p = 0; p < settings->NumPattern; p++) {
        for (auto j = 0; j < settings->configuration[settings->outputLayerIndex]; j++) {
            settings->trainingOutput[p][j] = (*out)[p][j];
        }
    }
    for (auto p = 0; p < settings->NumPattern; p++) {
        for (auto j = 0; j < settings->configuration[0]; j++) {
            settings->trainingInput[p][j] = (*in)[p][j];
        }
    }

    allocateMatrix(&Expected, settings->NumPattern + 1, settings->configuration[settings->outputLayerIndex] + 1, 0);
    memory += Expected.size;

    Layers = new Matrix[settings->numberOfLayers];
    for (auto i = 0; i < settings->numberOfLayers; i++) {
        allocateMatrix(&Layers[i], settings->NumPattern + 1, settings->configuration[i] + 1, 0);
        memory += Layers[i].size;
    }

    Weights = new Matrix[settings->numberOfWeights];
    for (auto i = 0; i < settings->numberOfWeights; i++) {
        allocateMatrix(&Weights[i], settings->configuration[i] + 1, settings->configuration[i + 1] + 1, -1.0, 1.0);
        memory += Weights[i].size;
    }

    Delta = new Array[settings->numberOfWeights];
    for (auto i = 0; i < settings->numberOfWeights; i++) {
        allocateMatrix(&Delta[i], settings->configuration[i + 1] + 1, 0);
        memory += Delta[i].size;
    }

    DeltaWeight = new Matrix[settings->numberOfWeights];
    for (auto i = 0; i < settings->numberOfWeights; i++) {
        allocateMatrix(&DeltaWeight[i], settings->configuration[i] + 1, settings->configuration[i + 1] + 1, 0);
        memory += DeltaWeight[i].size;
    }

    outputLayer = &Layers[settings->outputLayerIndex];
}

NeuralNetwork::~NeuralNetwork() {
    deallocate(&Expected);

    for (auto i = 0; i < settings->numberOfLayers; i++) {
        deallocate(&Layers[i]);
    }
    delete[] Layers;

    for (auto i = 0; i < settings->numberOfWeights; i++) {
        deallocate(&Weights[i]);
    }
    delete[] Weights;

    for (auto i = 0; i < settings->numberOfWeights; i++) {
        deallocate(&Delta[i]);
    }
    delete[] Delta;

    for (auto i = 0; i < settings->numberOfWeights; i++) {
        deallocate(&DeltaWeight[i]);
    }
    delete[] DeltaWeight;

    deallocate(&settings->trainingInput);
    deallocate(&settings->trainingOutput);
}

long NeuralNetwork::allocateMatrix(NetworkTypePtrPtr *matrix, int size1, int size2, double low, double high) {
    (*matrix) = new NetworkTypePtr[size1];
    long memory = size1 * sizeof(NetworkTypePtr);
    for (auto i = 0; i < size1; i++) {
        (*matrix)[i] = new NetworkType[size2];
        memory += size2 * sizeof(NetworkType);
        for (auto j = 0; j < size2; j++) {
            (*matrix)[i][j] = fRand(low, high);
        }
    }
    return memory;
}

long NeuralNetwork::allocateMatrix(NetworkTypePtrPtr *matrix, int size1, int size2, double value) {
    (*matrix) = new NetworkTypePtr[size1];
    long memory = size1 * sizeof(NetworkTypePtr);
    for (auto i = 0; i < size1; i++) {
        (*matrix)[i] = new NetworkType[size2];
        memory += size2 * sizeof(NetworkType);
        for (auto j = 0; j < size2; j++) {
            (*matrix)[i][j] = value;
        }
    }
    return size1 * size2 * sizeof(NetworkType);
}

void NeuralNetwork::allocateMatrix(MatrixPtr matrix, int size1, int size2, double low, double high) {
    matrix->rows = size1;
    matrix->cols = size2;
    matrix->size = allocateMatrix(&matrix->elements, size1, size2, low, high);
}

void NeuralNetwork::allocateMatrix(MatrixPtr matrix, int size1, int size2, double value) {
    matrix->rows = size1;
    matrix->cols = size2;
    matrix->size = allocateMatrix(&matrix->elements, size1, size2, value);
}

void NeuralNetwork::allocateMatrix(ArrayPtr matrix, int size1, double value) {
    matrix->elements = new NetworkType[size1];
    long memory = size1 * sizeof(NetworkType);
    matrix->number = size1;
    for (auto j = 0; j < size1; j++) {
        matrix->elements[j] = value;
    }
    matrix->size = memory;
}

long NeuralNetwork::allocateMatrix(NetworkTypePtr *matrix, int size1, double value) {
    (*matrix) = new NetworkType[size1];
    long memory = size1 * sizeof(NetworkType);
    for (auto j = 0; j < size1; j++) {
        (*matrix)[j] = value;
    }
    return memory;
}

void NeuralNetwork::deallocate(NetworkTypePtrPtr *matrix, int size) {
    for (auto i = 0; i < size; i++) {
        delete[] (*matrix)[i];
    }
    delete[] (*matrix);
    (*matrix) = NULL;
}

void NeuralNetwork::deallocate(NetworkTypePtr *matrix) {
    delete[] (*matrix);
    (*matrix) = NULL;
}

void NeuralNetwork::deallocate(MatrixPtr matrix) {
    deallocate(&matrix->elements, matrix->rows);
}

void NeuralNetwork::deallocate(ArrayPtr matrix) {
    delete[] matrix->elements;
    matrix->elements = NULL;
}

NetworkType NeuralNetwork::fRand(NetworkType fMin, NetworkType fMax) {
    NetworkType f = (NetworkType) rand() / (NetworkType) RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double NeuralNetwork::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x) {
    return x * (1 - x);
}

void NeuralNetwork::randomizeInput() {
    for (auto p = 1; p <= settings->NumPattern; p++) {
        settings->trainingSetOrder[p] = p;
    }
    for (auto p = 1; p <= settings->NumPattern; p++) {
        int np = p + ((NetworkType) rand() / ((NetworkType) RAND_MAX + 1)) * (settings->NumPattern + 1 - p);
        int op = settings->trainingSetOrder[p];
        settings->trainingSetOrder[p] = settings->trainingSetOrder[np];
        settings->trainingSetOrder[np] = op;
    }
}

void NeuralNetwork::output() {
    printf("\nNETWORK DATA - EPOCH %d\tError = %2.9Lf\n\nPat\t", epoch, Error);
    for (auto i = 1; i <= settings->configuration[0]; i++) {
        printf("Input%-4d\t", i);
    }
    for (auto k = 1; k <= settings->configuration[settings->outputLayerIndex]; k++) {
        printf("Target%-4d\tOutput%-4d\t", k, k);
    }
    for (auto p = 1; p <= settings->NumPattern; p++) {
        printf("\n%d\t", p);
        for (auto i = 1; i <= settings->configuration[0]; i++) {
            printf("%2.9Lf\t", Layers[0].elements[p][i]);
        }
        for (auto k = 1; k <= settings->configuration[settings->outputLayerIndex]; k++) {
            printf("%2.9Lf\t%2.9Lf\t", Expected.elements[p][k], outputLayer->elements[p][k]);
        }
    }
    cout << endl << endl << "Training Elapsed Time = " << chrono::duration_cast<chrono::microseconds>(tend - tbegin).count() / 1000000.0 << " seconds" << endl;
    printf("Training Memory Used = %ld bytes\n", memory);
    long numberOfNeurons = 0;
    for (auto k = 0; k < settings->numberOfLayers; k++) {
        numberOfNeurons += settings->configuration[k];
    }
    long numberOfConnections = 0;
    for (auto k = 0; k < settings->numberOfWeights; k++) {
        numberOfConnections += Weights[k].rows * Weights[k].cols;
    }
    printf("Neurons Used = %ld\n", numberOfNeurons);
    printf("Neuron Connections Used = %ld\n", numberOfConnections);
}

void NeuralNetwork::forward(int p) {
    for (auto k = 1; k <= settings->numberOfLayers - 1; k++) {
        for (auto j = 1; j <= settings->configuration[k]; j++) {
            NetworkType accumulate = Weights[k - 1].elements[0][j];
            for (auto i = 1; i <= settings->configuration[k - 1]; i++) {
                accumulate += Layers[k - 1].elements[p][i] * Weights[k - 1].elements[i][j];
            }
            Layers[k].elements[p][j] = sigmoid(accumulate);
        }
    }
}

void NeuralNetwork::computeError(int p) {
    for (auto i = 1; i <= settings->configuration[settings->outputLayerIndex]; i++) {
        NetworkType diff = (Expected.elements[p][i] - outputLayer->elements[p][i]);
        Error += 0.5 * diff * diff;
        Delta[settings->outputLayerIndex - 1].elements[i] = diff * sigmoidDerivative(outputLayer->elements[p][i]);
    }
    for (auto k = settings->numberOfLayers - 3; k >= 0; k--) {
        for (auto j = 1; j <= settings->configuration[k + 1]; j++) {
            NetworkType accumulate = 0.0;
            for (auto i = 1; i <= settings->configuration[k + 2]; i++) {
                accumulate += Weights[k + 1].elements[j][i] * Delta[k + 1].elements[i];
            }
            Delta[k].elements[j] = accumulate * sigmoidDerivative(Layers[k + 1].elements[p][j]);
        }
    }
}

void NeuralNetwork::backPropagate(int p) {
    for (auto k = settings->numberOfLayers - 2; k >= 0; k--) {
        for (auto j = 1; j <= settings->configuration[k + 1]; j++) {
            DeltaWeight[k].elements[0][j] = eta * Delta[k].elements[j] + alpha * DeltaWeight[k].elements[0][j];
            Weights[k].elements[0][j] += DeltaWeight[k].elements[0][j];
            for (auto i = 1; i <= settings->configuration[k]; i++) {
                DeltaWeight[k].elements[i][j] = eta * Layers[k].elements[p][i] * Delta[k].elements[j] + alpha * DeltaWeight[k].elements[i][j];
                Weights[k].elements[i][j] += DeltaWeight[k].elements[i][j];
            }
        }
    }
}

void NeuralNetwork::train() {
    tbegin = chrono::steady_clock::now();
    for (auto p = 0; p < settings->NumPattern; p++) {
        for (auto j = 0; j < settings->configuration[settings->outputLayerIndex]; j++) {
            Expected.elements[p + 1][j + 1] = settings->trainingOutput[p][j];
        }
    }
    for (auto p = 0; p < settings->NumPattern; p++) {
        for (auto j = 0; j < settings->configuration[0]; j++) {
            Layers[0].elements[p + 1][j + 1] = settings->trainingInput[p][j];
        }
    }
    for (epoch = 0; epoch < 100000; epoch++) {
        randomizeInput();
        Error = 0.0;
        for (auto np = 1; np <= settings->NumPattern; np++) {
            auto p = settings->trainingSetOrder[np];
            forward(p);
            computeError(p);
            backPropagate(p);
        }
        if (epoch % 100 == 0)
            printf("\nEpoch %-5d :   Error = %2.9Lf", epoch, Error);
        if (Error < 0.0004)
            break;
    }
    tend = chrono::steady_clock::now();
}
