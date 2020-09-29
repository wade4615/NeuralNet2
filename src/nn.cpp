//============================================================================
// Name        : nn.cpp
// Author      : Christopher D. Wade
// Version     : 1.0
// Copyright   : (c) 2020 Christopher D. Wade
// Description : Multihidden layer backpropagation net
//============================================================================
#include"nn.h"
using namespace std;

int main() {
    const int numPatterns = 4;
    Matrix trainingInput = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    Matrix trainingOutput = { { 0 }, { 1 }, { 1 }, { 0 } };
    int config[] = { 2, 2, 1 };

    NeuralNetwork net(numPatterns, &trainingInput, &trainingOutput, config, sizeof(config) / sizeof(int));
    net.train(100000);
    net.output();
    printf("\n\nGoodbye!\n\n");
    return 1;
}
