/*
 * Matrix.h
 *
 *  Created on: Sep 17, 2020
 *      Author: wade4
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>

using namespace std;

double fRand(double fMin, double fMax);

template<class T>
class Matrix {
        int m_rowSize;
        int m_colSize;
        T **elements;
    public:
        Matrix(int rowSize, int colSize, double low, double high) {
            m_rowSize = rowSize;
            m_colSize = colSize;
            elements = new T*[rowSize];
            for (auto i = 0; i < rowSize; i++) {
                elements[i] = new T[colSize];
                for (auto j = 0; j < m_colSize; j++) {
                    elements[i][j] = fRand(low, high);
                }
            }
        }

        Matrix(int rowSize, int colSize, double value) {
            m_rowSize = rowSize;
            m_colSize = colSize;
            elements = new T*[rowSize];
            for (auto i = 0; i < rowSize; i++) {
                elements[i] = new T[colSize];
                for (auto j = 0; j < m_colSize; j++) {
                    elements[i][j] = value;
                }
            }
        }

        Matrix(initializer_list<initializer_list<double>> list) {
            m_rowSize = (int) list.size();
            m_colSize = (int) (list.begin())->size();
            elements = new T*[m_rowSize];
            for (auto i = 0; i < m_rowSize; i++) {
                elements[i] = new T[m_colSize];
                for (auto j = 0; j < m_colSize; j++) {
                    elements[i][j] = ((list.begin() + i)->begin())[j];
                }
            }
        }

        virtual ~Matrix() {
            for (auto i = 0; i < m_rowSize + 1; i++) {
                delete[] elements[i];
            }
            delete[] elements;
        }

        T*& operator[](const int &index) {
            return elements[index];
        }

        int getRows() const {
            return m_rowSize;
        }

        int getColumns() const {
            return m_colSize;
        }
};
#endif /* MATRIX_H_ */
