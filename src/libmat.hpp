
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


#define PRECISION 4
#define WIDTH (PRECISION + 1)


using namespace sycl;


typedef std::vector<double> DVector;
typedef std::vector<int> IVector;
//typedef static_cast<unsigned long> stt_cast;



// C = alpha(A x B) + beta(C)
bool times_m(const Matrix &A, const Matrix &B, Matrix &C, double alpha=1.0, double beta = 0.0)
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
    for (long int i = 0; i < A.n_row; i++)
    {
        for (long int j = 0; j < A.n_col; j++)
        {
            std::cout << std::setw(WIDTH) << std::setprecision(PRECISION) << A.elem[i * A.n_col + j] << " ";
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


bool cholesky(Matrix &A, Matrix &L, double alpha = 1.0)
{

    //if (!A.is_symmetric())
    //{
    //    std::cout << "\nMatrix not symmetric\n";
    //    return 0;
    //}
    double sum;
    long int i, j, k;
    auto n = A.n_row;
    L.zeros();

    for (i = 0; i < n; i++)
    {
        for (j = 0; j <= i; j++)
        {
            sum = 0.0;
            if (i == j)//diagonals
            {
                for(k= 0; k < j; k++)
                    sum+= pow(L.elem[n*j+k], 2);
                L.elem[n*j+j] =  sqrt(A.elem[n*j+j]-sum);
                
            } else {
                //Evaluate L(i,j) using L(j, j)
                for(k = 0; k < j; k++)
                    sum+= L.elem[n*i+k] * L.elem[n*j+k];
                L.elem[n*i+j] = (A.elem[n*i+j] - sum)/L.elem[n*j+j];

            }
                
        }
    }
    
    return true;
}
    


/* 
####################################
        parallel Library
#################################### 
*/

// C = alpha(A ) +beta(B)
bool sum_m(sycl::queue &q, Matrix &A, Matrix &B, Matrix &C, double alpha = 1.0, double beta = 1.0){
    
    //if(!A.equal_dim(B) or !A.equal_dim(C)) return false;
    
    //Range of vectors to sum
    sycl::range<1> num_items{A.elem.size()};
    DVector scal_vector = {alpha, beta};
    
    //hold data shared between host and devices
    sycl::buffer a_buf(A.elem);
    sycl::buffer b_buf(B.elem);
    sycl::buffer c_buf(C.elem.data(), num_items);
    
    //scalar vector
    sycl::buffer scal_buf(scal_vector);
        
    
    //Submit data by lambda function, contains data access permition
     q.submit([&](handler &h) {
         sycl::accessor a(a_buf, h, read_only);
         sycl::accessor b(b_buf, h, read_only);
         sycl::accessor c(c_buf, h, write_only, no_init);
         
         //buff scalar
         sycl::accessor scal(scal_buf, h, read_only);
         
         h.parallel_for(num_items, [=](auto i){
             c[i] = scal[0] * a[i] + scal[1] * b[i];
         });
         
    });
    // Wait until compute tasks on GPU done
    q.wait();     
    
    return true;    
}


bool times_m(sycl::queue &q, const Matrix &A, const Matrix &B, Matrix &C, double alpha=1.0, double beta = 0.0)
{
    
    //if(!A.equal_dim(B) or !A.equal_dim(C)) return false;
    
    //Range of vectors to sum
    size_t size_col = A.elem.size()/A.n_row;
    size_t size_row = A.elem.size()/A.n_col;
    
    sycl::range<2> num_items{size_col, size_row};
    DVector scal_vector = {alpha, beta};
    
    int width = A.n_row;
    //hold data shared between host and devices, 
    //converted in 2D matrices: range(col, row)
    sycl::buffer<double,2> a_buf = {A.elem.data(), num_items};
    sycl::buffer<double,2> b_buf = {B.elem.data(), num_items};
    sycl::buffer<double,2> c_buf = {C.elem.data(), num_items};
    
    //create a sub-buffer with a_but converting in 2 dimensional buff
    
    //buffer b11{a_buf, id{0, 0}, range{size_row, size_col}};
        
    //hold scalar values 
    sycl::buffer scal_buf(scal_vector);
    
    q.submit([&](handler &h){
        sycl::accessor a(a_buf, h, read_only);
        sycl::accessor b(b_buf, h, read_only);
        sycl::accessor c(c_buf, h, write_only, no_init);
        h.parallel_for(num_items, [=](auto idx){
             int row = idx[0]; 
             int col = idx[1];
             double sum = 0.0f;  
             for(int i = 0; i < width; i++){
                 sum+= a[row][i] * b[i][col];
             }
             c[idx] = sum;
         });
             
    });
    // Wait until compute tasks on GPU done
    q.wait();
    
    //Submit data by lambda function, contains data access permition     
    
    return true;   
    
    
}

bool timesc_m(sycl::queue &q, double alpha, Matrix &A, Matrix &R)
{
    //check sizes..
    if(!A.equal_dim(R)) return false;
    //else...
    
    sycl::range<1> num_items{A.elem.size()};
    //todo: how to host only one number ? 
    DVector scal_vector = {alpha, 0.0};
    
    //hold data shared between host and devices
    sycl::buffer a_buf(A.elem);
    sycl::buffer r_buf(R.elem.data(), num_items);
    
    //scalar vector
    sycl::buffer scal_buf(scal_vector);
    
    
    //Submit data by lambda function, contains data access permition
    q.submit([&](handler &h) {
        sycl::accessor a(a_buf, h, read_only);
        sycl::accessor r(r_buf, h, write_only, no_init);
         
        //buff scalar
        sycl::accessor scal(scal_buf, h, read_only);
        

        h.parallel_for(num_items, [=](auto i){
             r[i] = scal[0] * a[i];
         });
         
    });
    // Wait until compute tasks on GPU done
    q.wait();     
    
    return true;
    
}

// possivel implementacao da inversa:
//A-1(i,j) = 1/detA * Cof(A(i,j))
//todo: det. Leibniz? 


//Cholesky parallel 
bool cholesky_v1(sycl::queue q, Matrix &A_m, Matrix &L_m, double alpha = 1.0){
    
    
    //check if there's a square Matrix declared: 
    // if (!A_m.square_m() || !L_m.square_m() || !A_m.is_symmetric())
    //{
    //    std::cout << "\nMatrix not symmetric\n";
    //    return 0;
    //} 
    // else ....
    
    size_t nr = A_m.n_row;
    
    //Set 2D buffers to host in devices
    sycl::range<2> num_items{nr, nr};
    
    sycl::buffer<double,2> A_buf = {A_m.elem.data(), num_items};
    sycl::buffer<double,2> L_buf = {L_m.elem.data(), num_items};
    
    //task A: copiar a matriz A em L - total de itens = nr*nr
    auto eA = q.submit([&](handler &h) {

        //declare accessor to buffers
        sycl::accessor a{A_buf, h};
        sycl::accessor l{L_buf, h};

        h.parallel_for(num_items, [=](auto idx){       
            int i = idx[0];
            int j = idx[1];
            if(i >= j) l[idx] = a[idx];
            else l[idx] = 0.0;
        });
    });
    q.wait();
    
    //print_m(L_m);
    
    //Algoritmo de Cholesky Incompleto, iterado ao longo de sua coluna
    for(int itr = 0; itr < nr; itr++)
    {
        //print_m(L_m);
        
        //Range for division:
        sycl::range<2>c_div{static_cast<unsigned long>(nr-(itr+1)),1};
        
        //sqrt in  l_kk of its diagonal entry
        L_m.elem[nr*itr + itr] = sqrt(L_m.elem[nr*itr + itr]);
        
        //task B: calcular a coluna (a(i,j) /=a(j,j)
        //for( i = k+1; i < n; i++)
        auto eB = q.submit([&](handler &h) {

            //declare accessor to buffers
            sycl::accessor l{L_buf, h};
            
            int k = itr;        
            
            h.parallel_for(c_div, [=](auto idx){
                int i = idx[0]+k+1;
                int j = idx[1]+k;
                l[i][j] /= l[k][k];                
            });            
        });
        eB.wait();
        
        
        //task C:Elimination B(i,j) -= B(i,k) * B.(j,k)
    
            for(int j = itr+1; j < nr; j++)
            {
                //Range for permutation:
            sycl::range<2> c_mod{static_cast<unsigned long>(nr-j),1};
                //for(int i = j_0; i < n; i++)
                auto eC = q.submit([&](handler &h) {
                    h.depends_on({eB});
                        
                    //declare accessor to buffers
                    sycl::accessor l{L_buf, h};

                    int k = itr;
                    int j_0 = j; 
                    
                    h.parallel_for(c_mod, [=](auto idx){     
                        int i = j_0 + idx[0];
                        
                        
                        l[i][j_0] -= l[i][k] * l[j_0][k] ;

                    });            

                });
                eC.wait();
              
                //print_m(L_m);
            }       
             
    }
    
    
    
    
    return true; 
}


// modificação da estrutura de paralelização
bool cholesky_v2(sycl::queue q, Matrix &A_m, Matrix &L_m, double alpha = 1.0){
    
    
    //check if there's a square Matrix declared: 
    // if (!A_m.square_m() || !L_m.square_m() || !A_m.is_symmetric())
    //{
    //    std::cout << "\nMatrix not symmetric\n";
    //    return 0;
    //} 
    // else ....
    
    size_t nr = A_m.n_row;
    
    //Set 2D buffers to host in devices
    sycl::range<2> num_items{nr, nr};
    
    sycl::buffer<double,2> A_buf = {A_m.elem.data(), num_items};
    sycl::buffer<double,2> L_buf = {L_m.elem.data(), num_items};
    
    //task A: copiar a matriz A em L - total de itens = nr*nr
    auto eA = q.submit([&](handler &h) {

        //declare accessor to buffers
        sycl::accessor a{A_buf, h};
        sycl::accessor l{L_buf, h};

        h.parallel_for(num_items, [=](auto idx){       
            int i = idx[0];
            int j = idx[1];
            if(i >= j) l[idx] = a[idx];
            else l[idx] = 0.0;
        });
    });
    q.wait();
    
    //print_m(L_m);
    
    //Algoritmo de Cholesky Incompleto, iterado ao longo de sua coluna
    for(int itr = 0; itr < nr; itr++)
    {
        //print_m(L_m);
        //Range for division:        
        //sqrt in  l_kk of its diagonal entry
        L_m.elem[nr*itr + itr] = sqrt(L_m.elem[nr*itr + itr]);
        
        //task B: calcular a coluna (a(i,j) /=a(j,j)
        //for( i = k+1; i < n; i++)
        auto eB = q.submit([&](handler &h) {

            //declare accessor to buffers
            sycl::accessor l{L_buf, h};
            
            int k = itr;        
            
           auto s_1 = h.parallel_for(
               sycl::range{static_cast<unsigned long>(nr-(itr+1)),1}, [=](auto idx){
                int i = idx[0]+k+1;
                int j = idx[1]+k;
                l[i][j] /= l[k][k]; 
            });
            
                     
            
            
        });
                
        //task C:Elimination B(i,j) -= B(i,k) * B.(j,k)
    
            for(int j = itr+1; j < nr; j++)
            {
                //Range for permutation:
            //sycl::range<2> c_mod{static_cast<unsigned long>(nr-j),1};
                //for(int i = j_0; i < n; i++)
                auto eC = q.submit([&](handler &h) {
                    h.depends_on({eB});
                        
                    //declare accessor to buffers
                    sycl::accessor l{L_buf, h};

                    int k = itr;
                    int j_0 = j; 
                    
                    h.parallel_for(sycl::range{nr-j, 1}, [=](auto idx){     
                        int i = j_0 + idx[0];
                        
                        
                        l[i][j_0] -= l[i][k] * l[j_0][k] ;

                    });            

                });
                eC.wait();
              
                //print_m(L_m);
            }       
             
    }
    
    
    
    
    return true; 
}


bool incomplete_cholesky(Matrix &A, Matrix &B)
{
    
    //if(!A.is_symmetric() || !A.is_square()) return false;

    //copy A and overwrite in B
    //B.copy_of(A);
    B.elem.assign(A.elem.begin(),A.elem.end());
    int nr = A.n_row;
    int i, j, k;
    for(k = 0; k < nr; k++)//for - sequencial
    {
        //calculate principal diag. 
        B.elem[nr*k+k] = sqrt(B.elem[nr*k+k]);

        for(i = k+1; i < nr; i++){
            //paralelo
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