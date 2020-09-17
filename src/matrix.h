/*
 * Matrix.h
 *
 *  Created on: Sep 17, 2020
 *      Author: wade4
 */

#ifndef MATRIX_H_
#define MATRIX_H_

typedef double NetworkType;
typedef NetworkType *NetworkTypePtr;
typedef NetworkTypePtr *NetworkTypePtrPtr;

using namespace std;

double fRand(double fMin, double fMax);

class Matrix {
        int m_rowSize;
        int m_colSize;
        double **elements;
    public:
        Matrix(int rowSize, int colSize, double low, double high);
        Matrix(int rowSize, int colSize, double value);
        Matrix(initializer_list<initializer_list<double>> list);
        virtual ~Matrix();
        NetworkTypePtr& operator[](const int &index);
        void print(char *text) const;
        int getRows() const;
        int getCols() const;
};
#endif /* MATRIX_H_ */
