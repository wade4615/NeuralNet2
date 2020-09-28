//============================================================================
// Name        : nn.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
#include"nn.h"

#define NUMPAT 4
#define NUMIN  2
#define NUMHID 2
#define NUMOUT 1

int epoch;
int numberOfExamples = 4;
int trainingSetOrder[4 + 1];
int configuration[] = { 2, 2, 1 };
int numberOfLayers = sizeof(configuration) / sizeof(int);
int NumberOfWeights = numberOfLayers - 1;
int NumPattern = 4;
int outputLayerIndex;
long memory = 0;

NetworkType trainingInput[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
NetworkType trainingOutput[4][1] = { { 0 }, { 1 }, { 1 }, { 0 } };
NetworkType Error, eta = 0.5, alpha = 0.9;

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

void initialize() {
    allocateMatrix(&Expected, numberOfExamples + 1, configuration[2] + 1, 0);
    memory += Expected.size;

    Layers = new Matrix[numberOfLayers];
    for (auto i = 0; i < numberOfLayers; i++) {
        allocateMatrix(&Layers[i], numberOfExamples + 1, configuration[i] + 1, 0);
        memory += Layers[i].size;
    }

    Weights = new Matrix[NumberOfWeights];
    for (auto i = 0; i < NumberOfWeights; i++) {
        allocateMatrix(&Weights[i], configuration[i] + 1, configuration[i + 1] + 1, -1.0, 1.0);
        memory += Weights[i].size;
    }

    Delta = new Array[NumberOfWeights];
    for (auto i = 0; i < NumberOfWeights; i++) {
        allocateMatrix(&Delta[i], configuration[i + 1] + 1, 0);
        memory += Delta[i].size;
    }

    DeltaWeight = new Matrix[NumberOfWeights];
    for (auto i = 0; i < NumberOfWeights; i++) {
        allocateMatrix(&DeltaWeight[i], configuration[i] + 1, configuration[i + 1] + 1, 0);
        memory += DeltaWeight[i].size;
    }
    outputLayerIndex = numberOfLayers - 1;
    outputLayer = &Layers[outputLayerIndex];
}

void shutDown() {
    deallocate(&Expected);

    for (auto i = 0; i < numberOfLayers; i++) {
        deallocate(&Layers[i]);
    }
    delete[] Layers;

    for (auto i = 0; i < NumberOfWeights; i++) {
        deallocate(&Weights[i]);
    }
    delete[] Weights;

    for (auto i = 0; i < NumberOfWeights; i++) {
        deallocate(&Delta[i]);
    }
    delete[] Delta;

    for (auto i = 0; i < NumberOfWeights; i++) {
        deallocate(&DeltaWeight[i]);
    }
    delete[] DeltaWeight;
}

void randomizeInput() {
    for (auto p = 1; p <= NumPattern; p++) {
        trainingSetOrder[p] = p;
    }
    for (auto p = 1; p <= NumPattern; p++) {
        int np = p + ((double) rand() / ((double) RAND_MAX + 1)) * (NumPattern + 1 - p);
        int op = trainingSetOrder[p];
        trainingSetOrder[p] = trainingSetOrder[np];
        trainingSetOrder[np] = op;
    }
}

void output() {
    printf("\nNETWORK DATA - EPOCH %d\tError = %f\n\nPat\t", epoch, Error);
    for (auto i = 1; i <= configuration[0]; i++) {
        printf("Input%-4d\t", i);
    }
    for (auto k = 1; k <= configuration[2]; k++) {
        printf("Target%-4d\tOutput%-4d\t", k, k);
    }
    for (auto p = 1; p <= NumPattern; p++) {
        printf("\n%d\t", p);
        for (auto i = 1; i <= configuration[0]; i++) {
            printf("%f\t", Layers[0].elements[p][i]);
        }
        for (auto k = 1; k <= configuration[outputLayerIndex]; k++) {
            printf("%f\t%f\t", Expected.elements[p][k], outputLayer->elements[p][k]);
        }
    }
    printf("\nmemory used %4.4f bytes", (double) memory);
}

void forward(int p) {
    for (auto k = 1; k <= numberOfLayers - 1; k++) {
        for (auto j = 1; j <= configuration[k]; j++) {
            NetworkType accumulate = Weights[k - 1].elements[0][j];
            for (auto i = 1; i <= configuration[k - 1]; i++) {
                accumulate += Layers[k - 1].elements[p][i] * Weights[k - 1].elements[i][j];
            }
            Layers[k].elements[p][j] = sigmoid(accumulate);
        }
    }
}

void computeError(int p) {
    for (auto i = 1; i <= configuration[outputLayerIndex]; i++) {
        NetworkType diff = (Expected.elements[p][i] - outputLayer->elements[p][i]);
        Error += 0.5 * diff * diff;
        Delta[outputLayerIndex - 1].elements[i] = diff * sigmoidDerivative(outputLayer->elements[p][i]);
    }
    int k = 0;
    for (auto j = 1; j <= configuration[1]; j++) {
        NetworkType accumulate = 0.0;
        for (auto i = 1; i <= configuration[2]; i++) {
            accumulate += Weights[k + 1].elements[j][i] * Delta[k + 1].elements[i];
        }
        Delta[k].elements[j] = accumulate * sigmoidDerivative(Layers[k + 1].elements[p][j]);
    }
}

void backPropagate(int p) {
    for (auto k = numberOfLayers - 2; k >= 0; k--) {
        for (auto j = 1; j <= configuration[k + 1]; j++) {
            DeltaWeight[k].elements[0][j] = eta * Delta[k].elements[j] + alpha * DeltaWeight[k].elements[0][j];
            Weights[k].elements[0][j] += DeltaWeight[k].elements[0][j];
            for (auto i = 1; i <= configuration[k]; i++) {
                DeltaWeight[k].elements[i][j] = eta * Layers[k].elements[p][i] * Delta[k].elements[j] + alpha * DeltaWeight[k].elements[i][j];
                Weights[k].elements[i][j] += DeltaWeight[k].elements[i][j];
            }
        }
    }
}

void train() {
    for (auto p = 0; p < NumPattern; p++) {
        for (auto j = 0; j < configuration[2]; j++) {
            Expected.elements[p + 1][j + 1] = trainingOutput[p][j];
        }
    }
    for (auto p = 0; p < NumPattern; p++) {
        for (auto j = 0; j < configuration[0]; j++) {
            Layers[0].elements[p + 1][j + 1] = trainingInput[p][j];
        }
    }
    for (epoch = 0; epoch < 100000; epoch++) {
        randomizeInput();
        Error = 0.0;
        for (auto np = 1; np <= NumPattern; np++) {
            auto p = trainingSetOrder[np];
            forward(p);
            computeError(p);
            backPropagate(p);
        }
//        if (epoch % 100 == 0)
//            printf("\nEpoch %-5d :   Error = %f", epoch, Error);
        if (Error < 0.0004)
            break;
    }
}

int main() {
    initialize();
    train();
    output();
    shutDown();
    printf("\n\nGoodbye!\n\n");
    return 1;
}
