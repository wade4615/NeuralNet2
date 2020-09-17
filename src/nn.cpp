/*
 ============================================================================
 Name        : nn.cpp
 Author      : Christopher D. Wade
 Version     : 1.0
 Copyright   : (c) 2020 Christopher D. Wade
 Description : A basic implementation of a back propagation neural network
 ============================================================================
 */
#include "nn.h"

using namespace std;

int main() {
    NeuralNetwork network(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE, NUMBER_OF_EXAMPLES);
    network.setTrainingData( { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 1, 1 } }, { { 0, 0 }, { 0, 0 }, { 0, 1 }, { 0, 1 }, { 0, 0 } });
    network.initializeInputHidden();
    network.initializeHiddenOutput();
    int epoch = network.train(1000000);
    network.printResults(epoch);
    printf("\n\nGoodbye!\n\n");
    return 1;
}

/*******************************************************************************/
