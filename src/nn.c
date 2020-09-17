/*
 ============================================================================
 Name        : nn.c
 Author      : cd wade
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
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

#define rando() ((double)rand()/((double)RAND_MAX+1))

int np, op, ranpat[NUMBER_OF_EXAMPLES + 1];
int NumPattern = NUMBER_OF_EXAMPLES, NumInput = INPUT_LAYER_SIZE, NumHidden = HIDDEN_LAYER_SIZE, NumOutput = OUTPUT_LAYER_SIZE;

double Input[NUMBER_OF_EXAMPLES + 1][INPUT_LAYER_SIZE + 1] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 1, 1 } };
double Target[NUMBER_OF_EXAMPLES + 1][OUTPUT_LAYER_SIZE + 1] = { { 0, 0 }, { 0, 0 }, { 0, 1 }, { 0, 1 }, { 0, 0 } };

//double sumHidden[NUMBER_OF_EXAMPLES+1][HIDDEN_LAYER_SIZE+1];
double inputHiddenWeights[INPUT_LAYER_SIZE + 1][HIDDEN_LAYER_SIZE + 1], hidden[NUMBER_OF_EXAMPLES + 1][HIDDEN_LAYER_SIZE + 1];
//double sumOutput[NUMBER_OF_EXAMPLES+1][OUTPUT_LAYER_SIZE+1];
double hiddenOutputWeights[HIDDEN_LAYER_SIZE + 1][OUTPUT_LAYER_SIZE + 1], output[NUMBER_OF_EXAMPLES + 1][OUTPUT_LAYER_SIZE + 1];

double deltaOutput[OUTPUT_LAYER_SIZE + 1], deltaHidden[HIDDEN_LAYER_SIZE + 1];
//double sumDeltaOutputWeights[HIDDEN_LAYER_SIZE+1];
double deltaWeightInputHidden[INPUT_LAYER_SIZE + 1][HIDDEN_LAYER_SIZE + 1], deltaWeightHiddenOutput[HIDDEN_LAYER_SIZE + 1][OUTPUT_LAYER_SIZE + 1];

double Error, eta = 0.5, alpha = 0.9;
//double smallwt = 0.5;

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

/* initialize WeightInputHidden and DeltaWeightInputHidden */
void initializeInputHidden() {
    for (int j = 1; j <= NumHidden; j++) {
        for (int i = 0; i <= NumInput; i++) {
            deltaWeightInputHidden[i][j] = 0.0;
            inputHiddenWeights[i][j] = fRand(-1.0, 1.0); //2.0 * ( rando() - 0.5) * smallwt;
        }
    }
}

/* initialize hiddenOutputWeights and DeltaWeightHiddenOutput */
void initializeHiddenOutput() {
    for (int k = 1; k <= NumOutput; k++) {
        for (int j = 0; j <= NumHidden; j++) {
            deltaWeightHiddenOutput[j][k] = 0.0;
            hiddenOutputWeights[j][k] = fRand(-1.0, 1.0); //2.0 * ( rando() - 0.5) * smallwt;
        }
    }
}

/* randomize order of training patterns */
void randomizeInput() {
    for (int p = 1; p <= NumPattern; p++) {
        ranpat[p] = p;
    }
    for (int p = 1; p <= NumPattern; p++) {
        np = p + rando() * (NumPattern + 1 - p);
        op = ranpat[p];
        ranpat[p] = ranpat[np];
        ranpat[np] = op;
    }
}

/* compute hidden unit activations */
void forwardInputHidden(int p) {
    for (int j = 1; j <= NumHidden; j++) {
        double activate = inputHiddenWeights[0][j];
        for (int i = 1; i <= NumInput; i++) {
            activate += Input[p][i] * inputHiddenWeights[i][j];
        }
        hidden[p][j] = sigmoid(activate);
    }
}

/* compute output unit activations */
void forwardHiddenOutput(int p) {
    for (int k = 1; k <= NumOutput; k++) {
        double activate = hiddenOutputWeights[0][k];
        for (int j = 1; j <= NumHidden; j++) {
            activate += hidden[p][j] * hiddenOutputWeights[j][k];
        }
        output[p][k] = sigmoid(activate); /* Sigmoidal Outputs */
    }
}

/* compute output unit errors */
void computeOutputError(int p) {
    for (int k = 1; k <= NumOutput; k++) {
        double diff = Target[p][k] - output[p][k];
        Error += 0.5 * diff * diff; /* SSE */
        deltaOutput[k] = diff * sigmoidDerivative(output[p][k]); /* Sigmoidal Outputs, SSE */
    }
}

/* 'back-propagate' errors to hidden layer */
void computeHiddenError(int p) {
    for (int j = 1; j <= NumHidden; j++) {
        double activate = 0.0;
        for (int k = 1; k <= NumOutput; k++) {
            activate += hiddenOutputWeights[j][k] * deltaOutput[k];
        }
        deltaHidden[j] = activate * sigmoidDerivative(hidden[p][j]); //hidden[p][j] * (1.0 - hidden[p][j]) ;
    }
}

/* update weights hiddenOutputWeights */
void backpropagateOutput(int p) {
    for (int k = 1; k <= NumOutput; k++) {
        deltaWeightHiddenOutput[0][k] = eta * deltaOutput[k] + alpha * deltaWeightHiddenOutput[0][k];
        hiddenOutputWeights[0][k] += deltaWeightHiddenOutput[0][k];
        for (int j = 1; j <= NumHidden; j++) {
            deltaWeightHiddenOutput[j][k] = eta * hidden[p][j] * deltaOutput[k] + alpha * deltaWeightHiddenOutput[j][k];
            hiddenOutputWeights[j][k] += deltaWeightHiddenOutput[j][k];
        }
    }
}

/* update weights inputHiddenWeights */
void backpropagateHidden(int p) {
    for (int j = 1; j <= NumHidden; j++) {
        deltaWeightInputHidden[0][j] = eta * deltaHidden[j] + alpha * deltaWeightInputHidden[0][j];
        inputHiddenWeights[0][j] += deltaWeightInputHidden[0][j];
        for (int i = 1; i <= NumInput; i++) {
            deltaWeightInputHidden[i][j] = eta * Input[p][i] * deltaHidden[j] + alpha * deltaWeightInputHidden[i][j];
            inputHiddenWeights[i][j] += deltaWeightInputHidden[i][j];
        }
    }
}

int main() {
    initializeInputHidden();
    initializeHiddenOutput();

    int numberOfEpochs = 1000000;
    int epoch;
    for (epoch = 0; epoch < numberOfEpochs; epoch++) { /* iterate weight updates */
        randomizeInput();
        Error = 0.0;
        for (np = 1; np <= NumPattern; np++) { /* repeat for all the training patterns */
            int p = ranpat[np];

            forwardInputHidden(p);
            forwardHiddenOutput(p);

            computeOutputError(p);
            computeHiddenError(p);

            backpropagateOutput(p);
            backpropagateHidden(p);
        }
        if (epoch % 100000 == 0) {
            printf("\nEpoch %-5d :   Error = %f", epoch, Error);
        }
        if (Error < 0.0004)
            break; /* stop learning when 'near enough' */
    }

    printf("\nEpoch %-5d :   Error = %f", epoch, Error);
    printf("\n\nNETWORK DATA - EPOCH %d\n\nPat\t", numberOfEpochs); /* print network outputs */
    for (int i = 1; i <= NumInput; i++) {
        printf("Input%-4d\t", i);
    }
    for (int k = 1; k <= NumOutput; k++) {
        printf("Target%-4d\tOutput%-4d\t", k, k);
    }
    for (int p = 1; p <= NumPattern; p++) {
        printf("\n%d\t", p);
        for (int i = 1; i <= NumInput; i++) {
            printf("%f\t", Input[p][i]);
        }
        for (int k = 1; k <= NumOutput; k++) {
            printf("%f\t%f\t", Target[p][k], output[p][k]);
        }
    }
    printf("\n\nGoodbye!\n\n");
    return 1;
}

/*******************************************************************************/
