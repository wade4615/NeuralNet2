/*
 ============================================================================
 Name        : nn.cpp
 Author      : Christopher D. Wade
 Version     : 1.0
 Copyright   : (c) 2020 Christopher D. Wade
 Description : A basic implementation of a back propagation nerual network
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>

#define NUMBER_OF_EXAMPLES 4
#define INPUT_LAYER_SIZE  2
#define HIDDEN_LAYER_SIZE 2
#define OUTPUT_LAYER_SIZE 1

int *trainingSetOrder;
int numTrainingSets = NUMBER_OF_EXAMPLES, inputSize = INPUT_LAYER_SIZE, middleSize = HIDDEN_LAYER_SIZE, outputSize = OUTPUT_LAYER_SIZE;

double trainingInput[NUMBER_OF_EXAMPLES + 1][INPUT_LAYER_SIZE + 1] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 1, 1 } };
double trainingOutput[NUMBER_OF_EXAMPLES + 1][OUTPUT_LAYER_SIZE + 1] = { { 0, 0 }, { 0, 0 }, { 0, 1 }, { 0, 1 }, { 0, 0 } };

double inputMiddleWeights[INPUT_LAYER_SIZE + 1][HIDDEN_LAYER_SIZE + 1], middleLayer[NUMBER_OF_EXAMPLES + 1][HIDDEN_LAYER_SIZE + 1];
double middleOutputWeights[HIDDEN_LAYER_SIZE + 1][OUTPUT_LAYER_SIZE + 1], outputLayer[NUMBER_OF_EXAMPLES + 1][OUTPUT_LAYER_SIZE + 1];

double outputLayerDelta[OUTPUT_LAYER_SIZE + 1], middleLayerDelta[HIDDEN_LAYER_SIZE + 1];
double deltaWeightInputMiddle[INPUT_LAYER_SIZE + 1][HIDDEN_LAYER_SIZE + 1], deltaWeightMiddleOutput[HIDDEN_LAYER_SIZE + 1][OUTPUT_LAYER_SIZE + 1];

double Error, eta = 0.5, alpha = 0.9;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1 - x);
}

double fRand(double fMin, double fMax) {
    double f = (double) rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
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

/* initialize WeightInputHidden and DeltaWeightInputHidden */
void initializeInputHidden() {
    for (auto j = 1; j <= middleSize; j++) {
        for (auto i = 0; i <= inputSize; i++) {
            deltaWeightInputMiddle[i][j] = 0.0;
            inputMiddleWeights[i][j] = fRand(-1.0, 1.0);
        }
    }
}

/* initialize hiddenOutputWeights and DeltaWeightHiddenOutput */
void initializeHiddenOutput() {
    for (auto k = 1; k <= outputSize; k++) {
        for (auto j = 0; j <= middleSize; j++) {
            deltaWeightMiddleOutput[j][k] = 0.0;
            middleOutputWeights[j][k] = fRand(-1.0, 1.0);
        }
    }
}

/* randomize order of training patterns */
void randomizeInput() {
    for (auto p = 0; p < numTrainingSets; p++) {
        trainingSetOrder[p] = p;
    }
    shuffle(trainingSetOrder, numTrainingSets);
}

/* compute hidden unit activations */
void forwardInputHidden(int p) {
    for (auto j = 1; j <= middleSize; j++) {
        double activate = inputMiddleWeights[0][j];
        for (auto i = 1; i <= inputSize; i++) {
            activate += trainingInput[p + 1][i] * inputMiddleWeights[i][j];
        }
        middleLayer[p][j] = sigmoid(activate);
    }
}

/* compute output unit activations */
void forwardHiddenOutput(int p) {
    for (auto k = 1; k <= outputSize; k++) {
        double activate = middleOutputWeights[0][k];
        for (auto j = 1; j <= middleSize; j++) {
            activate += middleLayer[p][j] * middleOutputWeights[j][k];
        }
        outputLayer[p][k] = sigmoid(activate); /* Sigmoidal Outputs */
    }
}

/* compute output unit errors */
void computeOutputError(int p) {
    for (auto k = 1; k <= outputSize; k++) {
        double diff = trainingOutput[p][k] - outputLayer[p][k];
        Error += 0.5 * diff * diff; /* SSE */
        outputLayerDelta[k] = diff * sigmoidDerivative(outputLayer[p][k]); /* Sigmoidal Outputs, SSE */
    }
}

/* 'back-propagate' errors to hidden layer */
void computeHiddenError(int p) {
    for (auto j = 1; j <= middleSize; j++) {
        double activate = 0.0;
        for (auto k = 1; k <= outputSize; k++) {
            activate += middleOutputWeights[j][k] * outputLayerDelta[k];
        }
        middleLayerDelta[j] = activate * sigmoidDerivative(middleLayer[p][j]); //hidden[p][j] * (1.0 - hidden[p][j]) ;
    }
}

/* update weights hiddenOutputWeights */
void backpropagateOutput(int p) {
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
void backpropagateHidden(int p) {
    for (auto j = 1; j <= middleSize; j++) {
        deltaWeightInputMiddle[0][j] = eta * middleLayerDelta[j] + alpha * deltaWeightInputMiddle[0][j];
        inputMiddleWeights[0][j] += deltaWeightInputMiddle[0][j];
        for (auto i = 1; i <= inputSize; i++) {
            deltaWeightInputMiddle[i][j] = eta * trainingInput[p + 1][i] * middleLayerDelta[j] + alpha * deltaWeightInputMiddle[i][j];
            inputMiddleWeights[i][j] += deltaWeightInputMiddle[i][j];
        }
    }
}

void printResults(int epoch) {
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
            printf("%f\t", trainingInput[p + 1][i]);
        }
        for (auto k = 1; k <= outputSize; k++) {
            printf("%f\t%f\t", trainingOutput[p][k], outputLayer[p][k]);
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

int main() {
    trainingSetOrder = new int[NUMBER_OF_EXAMPLES];
    initializeInputHidden();
    initializeHiddenOutput();
    int epoch = train(1000000);
    printResults(epoch);
    printf("\n\nGoodbye!\n\n");
    delete[] trainingSetOrder;
    return 1;
}

/*******************************************************************************/
