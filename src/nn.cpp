//============================================================================
// Name        : nn.cpp
// Author      : Christopher D. Wade
// Version     : 1.0
// Copyright   : (c) 2020 Christopher D. Wade
// Description : Multihidden layer backpropagation net
//============================================================================
#include"nn.h"

int epoch;
long memory = 0;
long double Error;

int *trainingSetOrder;
int outputLayerIndex;
NetworkType eta = 0.5, alpha = 0.9;

NetworkType trainingInput[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
NetworkType trainingOutput[4][1] = { { 0 }, { 1 }, { 1 }, { 0 } };

MatrixPtr Layers;
MatrixPtr Weights;
Matrix Expected; //[NUMPAT + 1][NUMOUT + 1];
ArrayPtr Delta;
MatrixPtr DeltaWeight;
MatrixPtr outputLayer;

NetworkType fRand(NetworkType fMin, NetworkType fMax) {
    NetworkType f = (NetworkType) rand() / (NetworkType) RAND_MAX;
    return fMin + f * (fMax - fMin);
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1 - x);
}

void initialize(Settings settings) {
    trainingSetOrder = new int[settings.NumPattern + 1];

    outputLayerIndex = settings.numberOfLayers - 1;

    allocateMatrix(&Expected, settings.NumPattern + 1, settings.configuration[outputLayerIndex] + 1, 0);
    memory += Expected.size;

    Layers = new Matrix[settings.numberOfLayers];
    for (auto i = 0; i < settings.numberOfLayers; i++) {
        allocateMatrix(&Layers[i], settings.NumPattern + 1, settings.configuration[i] + 1, 0);
        memory += Layers[i].size;
    }

    Weights = new Matrix[settings.numberOfWeights];
    for (auto i = 0; i < settings.numberOfWeights; i++) {
        allocateMatrix(&Weights[i], settings.configuration[i] + 1, settings.configuration[i + 1] + 1, -1.0, 1.0);
        memory += Weights[i].size;
    }

    Delta = new Array[settings.numberOfWeights];
    for (auto i = 0; i < settings.numberOfWeights; i++) {
        allocateMatrix(&Delta[i], settings.configuration[i + 1] + 1, 0);
        memory += Delta[i].size;
    }

    DeltaWeight = new Matrix[settings.numberOfWeights];
    for (auto i = 0; i < settings.numberOfWeights; i++) {
        allocateMatrix(&DeltaWeight[i], settings.configuration[i] + 1, settings.configuration[i + 1] + 1, 0);
        memory += DeltaWeight[i].size;
    }

    outputLayer = &Layers[outputLayerIndex];
}

void shutDown(Settings settings) {
    deallocate(&Expected);

    for (auto i = 0; i < settings.numberOfLayers; i++) {
        deallocate(&Layers[i]);
    }
    delete[] Layers;

    for (auto i = 0; i < settings.numberOfWeights; i++) {
        deallocate(&Weights[i]);
    }
    delete[] Weights;

    for (auto i = 0; i < settings.numberOfWeights; i++) {
        deallocate(&Delta[i]);
    }
    delete[] Delta;

    for (auto i = 0; i < settings.numberOfWeights; i++) {
        deallocate(&DeltaWeight[i]);
    }
    delete[] DeltaWeight;
}

void randomizeInput(Settings settings) {
    for (auto p = 1; p <= settings.NumPattern; p++) {
        trainingSetOrder[p] = p;
    }
    for (auto p = 1; p <= settings.NumPattern; p++) {
        int np = p + ((NetworkType) rand() / ((NetworkType) RAND_MAX + 1)) * (settings.NumPattern + 1 - p);
        int op = trainingSetOrder[p];
        trainingSetOrder[p] = trainingSetOrder[np];
        trainingSetOrder[np] = op;
    }
}

void output(Settings settings) {
    printf("\nNETWORK DATA - EPOCH %d\tError = %2.9Lf\n\nPat\t", epoch, Error);
    for (auto i = 1; i <= settings.configuration[0]; i++) {
        printf("Input%-4d\t", i);
    }
    for (auto k = 1; k <= settings.configuration[outputLayerIndex]; k++) {
        printf("Target%-4d\tOutput%-4d\t", k, k);
    }
    for (auto p = 1; p <= settings.NumPattern; p++) {
        printf("\n%d\t", p);
        for (auto i = 1; i <= settings.configuration[0]; i++) {
            printf("%2.9Lf\t", Layers[0].elements[p][i]);
        }
        for (auto k = 1; k <= settings.configuration[outputLayerIndex]; k++) {
            printf("%2.9Lf\t%2.9Lf\t", Expected.elements[p][k], outputLayer->elements[p][k]);
        }
    }
    printf("\nmemory used %4.4f bytes", (double) memory);
}

void forward(Settings settings, int p) {
    for (auto k = 1; k <= settings.numberOfLayers - 1; k++) {
        for (auto j = 1; j <= settings.configuration[k]; j++) {
            NetworkType accumulate = Weights[k - 1].elements[0][j];
            for (auto i = 1; i <= settings.configuration[k - 1]; i++) {
                accumulate += Layers[k - 1].elements[p][i] * Weights[k - 1].elements[i][j];
            }
            Layers[k].elements[p][j] = sigmoid(accumulate);
        }
    }
}

void computeError(Settings settings, int p) {
    for (auto i = 1; i <= settings.configuration[outputLayerIndex]; i++) {
        NetworkType diff = (Expected.elements[p][i] - outputLayer->elements[p][i]);
        Error += 0.5 * diff * diff;
        Delta[outputLayerIndex - 1].elements[i] = diff * sigmoidDerivative(outputLayer->elements[p][i]);
    }
    for (auto k = settings.numberOfLayers - 3; k >= 0; k--) {
        for (auto j = 1; j <= settings.configuration[k + 1]; j++) {
            NetworkType accumulate = 0.0;
            for (auto i = 1; i <= settings.configuration[k + 2]; i++) {
                accumulate += Weights[k + 1].elements[j][i] * Delta[k + 1].elements[i];
            }
            Delta[k].elements[j] = accumulate * sigmoidDerivative(Layers[k + 1].elements[p][j]);
        }
    }
}

void backPropagate(Settings settings, int p) {
    for (auto k = settings.numberOfLayers - 2; k >= 0; k--) {
        for (auto j = 1; j <= settings.configuration[k + 1]; j++) {
            DeltaWeight[k].elements[0][j] = eta * Delta[k].elements[j] + alpha * DeltaWeight[k].elements[0][j];
            Weights[k].elements[0][j] += DeltaWeight[k].elements[0][j];
            for (auto i = 1; i <= settings.configuration[k]; i++) {
                DeltaWeight[k].elements[i][j] = eta * Layers[k].elements[p][i] * Delta[k].elements[j] + alpha * DeltaWeight[k].elements[i][j];
                Weights[k].elements[i][j] += DeltaWeight[k].elements[i][j];
            }
        }
    }
}

void train(Settings settings) {
    for (auto p = 0; p < settings.NumPattern; p++) {
        for (auto j = 0; j < settings.configuration[outputLayerIndex]; j++) {
            Expected.elements[p + 1][j + 1] = trainingOutput[p][j];
        }
    }
    for (auto p = 0; p < settings.NumPattern; p++) {
        for (auto j = 0; j < settings.configuration[0]; j++) {
            Layers[0].elements[p + 1][j + 1] = trainingInput[p][j];
        }
    }
    for (epoch = 0; epoch < 100000; epoch++) {
        randomizeInput(settings);
        Error = 0.0;
        for (auto np = 1; np <= settings.NumPattern; np++) {
            auto p = trainingSetOrder[np];
            forward(settings, p);
            computeError(settings, p);
            backPropagate(settings, p);
        }
        if (epoch % 100 == 0)
            printf("\nEpoch %-5d :   Error = %2.9Lf", epoch, Error);
        if (Error < 0.0004)
            break;
    }
}

NetworkType& Array::operator[](int index) {
    return elements[index];
}

int main() {
    int config[] = { 2, 2, 2, 1 };

    Settings settings;
    settings.numberOfLayers = sizeof(config) / sizeof(int);
    settings.numberOfWeights = settings.numberOfLayers - 1;
    allocateMatrix(&settings.configuration, settings.numberOfLayers, 0);
    for (auto i = 0; i < settings.numberOfLayers; i++) {
        settings.configuration[i] = config[i];
    }
    settings.NumPattern = 4;

    initialize(settings);
    train(settings);
    output(settings);
    shutDown(settings);
    printf("\n\nGoodbye!\n\n");
    return 1;
}
