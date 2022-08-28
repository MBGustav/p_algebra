#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <random>

#include "defines.h"
#include "structs.hpp"

#ifndef LIBMAT_HPP
#define LIBMAT_HPP


// C = alpha(A x B) + beta(C)
bool times_m( Matrix &A,  Matrix &B, Matrix &C, double alpha=1.0, double beta = 0.0)
{

    // check sizes:
    if (A.n_col != B.n_row)
    {
        return false;
    }
    double sum = 0.0;
    for (long int i = 0; i < A.n_row; i++)
    {

        for (long int j = 0; j < B.n_col; j++)
        {
            sum = 0;
            for (long int k = 0; k < B.n_row; k++)
            {
                sum += A.elem[i * A.n_col + k] * B.elem[k * B.n_col + j];
            }
            C.elem[B.n_col * i + j]= alpha*sum+beta*C.elem[B.n_col * i + j];
        }
    }
    return true;
}

void equalpart_m(Matrix &A, int i_0, int j_0, Matrix &B) {
        int nrA, ncA, nrB, ncB;
        int nr_f, nc_f;
        int i_A,j_A, i_B, j_B;
        

        nrA = A.n_row;
        ncA = A.n_col;
        nrB = B.n_row;
        ncB = B.n_col;

        nr_f = (i_0 + nrB);  // final row
        nc_f = (j_0 + ncB);  // final column

        for (i_A = i_0, i_B =0; i_A < nr_f; i_A++, i_B++) {
                for (j_A = j_0, j_B = 0; j_A < nc_f; j_A++, j_B++) {
                        A.elem[i_A*ncA + j_A] = B.elem[i_B*ncB + j_B];
                }
        }

}

void getpart_m(Matrix &A, int i_0, int j_0, Matrix &B) {

        int nrA, ncA, nrB, ncB;
        int nr_f, nc_f;
        int i_A,j_A, i_B, j_B;
        double *pA, *pB;

        nrA = A.n_row;
        ncA = A.n_col;
        nrB = B.n_row;
        ncB = B.n_col;

        nr_f = (i_0 + nrB);  // final row
        nc_f = (j_0 + ncB);  // final column

        for (i_A = i_0, i_B =0; i_A < nr_f; i_A++, i_B++) {
                for (j_A = j_0, j_B = 0; j_A < nc_f; j_A++, j_B++) {
                         B.elem[i_B*ncB + j_B] = A.elem[i_A*ncA + j_A];
                }
        }

}

bool timesc_m(double val, Matrix &A, Matrix &R)
{
    if (A.size() != R.size())
        return false;

    for (long int i = 0; i < A.size(); i++)
    {
        R.elem[i] = val * A.elem[i];
    }
    return true;
}

void print_m(const Matrix &A)
{



    for ( int i = 0; i < A.n_row; i++)
    {
        for ( int j = 0; j < A.n_col; j++)
        {
            std::cout << A.at(i,j) << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    
}




// C = alpha * A + beta * B
bool sum_m(Matrix &A, Matrix &B, Matrix &C, double alpha = 1.0, double beta = 1.0)
{
    // check sizes
    if (A.size() != B.size() || A.size() != C.size())
        return false;
    
    
    for (long int i = 0; i < A.size(); i++)
    {
        C.elem[i] = beta * B.elem[i] + alpha * A.elem[i];
    }
    return true;
}

bool less_m(Matrix &A, Matrix &B, Matrix &C, double alpha=1.0, double beta =1.0){
    return sum_m(A,B,C, 1.0, -1.0);
}

bool transp_m(const Matrix &A, Matrix &R)
{
    if (A.n_row != R.n_col || A.n_col != R.n_row)
    {
        std::cout << "Not transposed!\n";
        return false;
    }

    for (long int i = 0; i < A.n_row; i++)
    {
        for (long int j = 0; j < A.n_col; j++)
        {
            // C[i,j] = A[j,i]
            R.elem[A.n_row * j + i] = A.elem[A.n_col * i + j];
        }
    }
    return 1;
}

short int inv_m(Matrix &A, Matrix &R)
{
    if (A.n_row != R.n_row || A.n_col != R.n_col)
    {
        std::cout << "Different dimension !\n ";
        return false;
    }

    //---------------------------- Partial pivoting --------------------------------
    long int i, j, k, cont;
    long int idx2, mem, flag;
    double sum;
    DVector b(A.n_row);
    DVector x(A.n_row);
    IVector idx(A.n_row);

    // Copy matrix - auxiliar
    Matrix *a = new Matrix(A.n_row, A.n_col);
    a->elem = A.elem;

    for (k = 0; k < A.n_row; k++)
        idx[k] = k;

    for (i = 0; i < A.n_row; i++)
    {
        j = i;
        idx2 = i;
        if (a->elem[A.n_col * i + j] == 0)
        {
            flag = 1;
            for (k = i + 1; k < A.n_row; k++)
            {
                if (fabs(a->elem[A.n_col * k + j]) >= TINY && flag == 1)
                {
                    mem = idx[i];
                    idx[i] = idx[k];
                    idx[k] = mem;
                    idx2 = k;
                    flag = 0;
                }
            }
            if (flag == 1)
            {
                for (k = 0; k < A.n_row; k++)
                {
                    if (fabs(a->elem[A.n_col * k + j]) > TINY && fabs(a->elem[A.n_col * i + k]) > TINY)
                    {
                        mem = idx[i];
                        idx[i] = idx[k];
                        idx[k] = mem;
                        idx2 = k;
                        flag = 0;
                    }
                }
            }
            if (idx2 == i)
            {
                printf("\n Singular matrix \n \n");
                a->elem[A.n_col * i + j] = TINY;
            }
            for (k = 0; k < A.n_row; k++)
            {
                mem = a->elem[A.n_col * i + k];
                a->elem[A.n_col * i + k] = a->elem[A.n_col * idx2 + k];
                a->elem[A.n_col * idx2 + k] = mem;
            }
        }
    }

    //------------------- Crout's algorithm for LU Decomposition -------------------
    for (j = 0; j < A.n_row; j++)
    {
        for (i = 0; i < A.n_row; i++)
        {
            if (i < j | i == j)
            {
                sum = a->elem[A.n_col * i + j];
                for (k = 0; k < i; k++)
                {
                    sum = sum - a->elem[A.n_col * i + k] * a->elem[A.n_col * k + j];
                }
                a->elem[A.n_col * i + j] = sum;
            }
            if (i > j)
            {
                sum = a->elem[A.n_col * i + j];
                for (k = 0; k < j; k++)
                {
                    sum = sum - a->elem[A.n_col * i + k] * a->elem[A.n_col * k + j];
                }
                a->elem[A.n_col * i + j] = sum / a->elem[A.n_col * j + j];
            }
        }
    }
    //---------------------------- Forward substituion -----------------------------
    for (k = 0; k < A.n_row; k++)
    {
        for (cont = 0; cont < A.n_row; cont++)
        {
            b[cont] = 0;
        }
        b[k] = 1;
        for (i = 0; i < A.n_row; i++)
        {
            sum = b[i];
            for (j = 0; j < i; j++)
            {
                sum = sum - a->elem[A.n_col * i + j] * x[j];
            }
            x[i] = sum;
        }
        //---------------------------- Backward substituion ----------------------------
        for (i = (A.n_row - 1); i >= 0; i--)
        {
            sum = x[i];
            for (j = i + 1; j < A.n_row; j++)
            {
                sum = sum - a->elem[A.n_col * i + j] * x[j];
            }
            x[i] = sum / a->elem[A.n_col * i + i];
        }
        for (cont = 0; cont < A.n_row; cont++)
        {
            R.elem[A.n_col * cont + idx[k]] = x[cont];
        }
    }
    delete a;
    b.clear();
    x.clear();
    idx.clear();

    return true;
}

void givens_m(Matrix &A, Matrix &R)
{

    int i, j, k, cont;
    int b, c, rho;
    double a; //alteração de "a" para double
    short int flag;
    long int i2, j2;
    Matrix *Theta = new Matrix(A.n_row, A.n_row);

    for (i = 0; i < A.n_row; i++)
    {
        for (j = A.n_col - 1; j >= i + 1; j--)
        {
            b = A.elem[A.n_col * i + j];
            flag = 0;
            for (cont = i; cont < j; cont++)
            {
                a = A.elem[A.n_col * i + cont];
                if (fabs(a) >= TINY)
                {
                    flag = 1;
                    break;
                }
            }
            if (flag == 0)
            {
                a = TINY;
                printf("\n a = 0 \n");
            }
            rho = b / a;
            for (i2 = 0; i2 < A.n_col; i2++)
            {
                for (j2 = 0; j2 < A.n_col; j2++)
                {
                    if (i2 == j2)
                    {
                        Theta->elem[A.n_col * i2 + j2] = 1;
                    }
                    else
                    {
                        Theta->elem[A.n_col * i2 + j2] = 0;
                    }
                }
            }
            c = 1 / sqrt(1 + rho * rho);
            Theta->elem[A.n_col * cont + cont] = c;
            Theta->elem[A.n_col * cont + j] = -rho * c;
            Theta->elem[A.n_col * j + cont] = rho * c;
            Theta->elem[A.n_col * j + j] = c;

            times_m(A, *Theta, R);
        }
    }
    delete Theta;
}


bool cholesky(Matrix *A, Matrix *L, double alpha = 1.0)
{

    //if (!A.is_symmetric())
    //{
    //    std::cout << "\nMatrix not symmetric\n";
    //    return 0;
    //}
    double sum;
    long int i, j, k;
    auto n = A->n_row;
    L->zeros();

    for (i = 0; i < n; i++)
    {
        for (j = 0; j <= i; j++)
        {
            sum = 0.0;
            if (i == j)//diagonals
            {
                for(k= 0; k < j; k++)
                    sum+= pow(L->elem[n*j+k], 2);
                L->elem[n*j+j] =  sqrt(alpha * A->elem[n*j+j]-sum);
                
            } else {
                //Evaluate L(i,j) using L(j, j)
                for(k = 0; k < j; k++)
                    sum+= L->elem[n*i+k] * L->elem[n*j+k];
                L->elem[n*i+j] = (alpha * A->elem[n*i+j] - sum)/L->elem[n*j+j];

            }
                
        }
    }
    
    return true;
}

bool incomplete_cholesky(Matrix &A, Matrix &B)
{
    
    if(!A.is_symmetric() || !A.square_m()) return false;

    B.elem.assign(A.elem.begin(),A.elem.end());
    int nr = A.n_row;
    int i, j, k;
    for(k = 0; k < nr; k++)
    {
        //calculate main diag. 
        B.elem[nr*k+k] = sqrt(B.elem[nr*k+k]);

        for(i = k+1; i < nr; i++){
            
            if(B.elem[nr*i+k]!=0) B.elem[nr*i+k] /=  B.elem[nr*k+k]; 
        }
    
        //Elimination    
        for(j = k+1; j < nr; j++)
        {
            for(i = j; i < nr; i++)
            {
            
                // std::cout << "U[" << i << ", " << j <<"]=";
                // std::cout << "U[" << i << ", " << k <<"] * ";
                // std::cout << "U[" << j << ", " << k <<"] = ";
                // std::cout << B.elem[nr*i+j] << "="<< B.elem[nr*i+k] << "*" << B.elem[nr*j+k] << "\n";
                B.elem[nr*i+j] -=  B.elem[nr*i+k] * B.elem[nr*j+k];
            }
        }

    }
	
    for(i = 0; i < nr; i++)
    {
        for(j = i+1; j < nr; j++)
        {
            B.elem[nr*i+j] = 0.0;
            
        }
    }
    

    return true;
}	


void get_inversible_matrix(Matrix &A, double medium_val = 1000.0)
{
    //Generate a upper-triangular matrix, then multiply by transpose 
    // A = B x B_t (read-write function)

    std::default_random_engine generator;
	std::normal_distribution<float> distribution(0.0, 1.0);

    A.zeros();
    
    int nr = A.n_row, nc = A.n_col; 
    for(auto i = 0; i < nr; i++)
    {
        for(auto j = i; j < nc; j++)
        {
            A.elem[nr*i+j] = medium_val * distribution(generator);
        }
    }

    Matrix *A_t = new Matrix(nr, nc);
    transp_m(A, *A_t);   
    
    times_m(A,*A_t, A);

    delete A_t;
    
}



#endif// LIBMAT_HPP