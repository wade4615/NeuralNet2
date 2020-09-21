/*
 * neuralNetwork.cpp
 *
 *  Created on: Sep 17, 2020
 *      Author: wade4
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
//#include "matrix.h"
//#include "neuralNetwork.h"

double fRand(double fMin, double fMax) {
    double f = (double) rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}
