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

int main() {
    Matrix<double> trainingInput = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
    Matrix<double> trainingOutput = { { 0 }, { 1 }, { 1 }, { 0 } };

    NeuralNetwork<double> network(2, 4, 1, 4, 1);

    network.setTrainingData(&trainingInput, &trainingOutput);

    int epoch = network.train(1000000);

    network.printResults(epoch);

    printf("\n\nGoodbye!\n\n");
    return 1;
}
