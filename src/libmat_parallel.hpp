#include <iostream>
#include <vector>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <random>
#include <CL/sycl.hpp>


#include "libmat.hpp"
#include "defines.h"
#include "structs.hpp"

// #include "libmat.hpp"

#ifndef LIBMAT_PARALLEL_HPP
#define LIBMAT_PARALLEL_HPP


#define PRECISION 4
#define WIDTH (PRECISION + 1)


using namespace sycl;


typedef std::vector<double> DVector;

// C = alpha(A ) +beta(B)
bool sum_m(sycl::queue &q, Matrix &A, Matrix &B, Matrix &C, double alpha = 1.0, double beta = 1.0){
        
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

//Cholesky parallel 
bool cholesky_v1(sycl::queue &q, Matrix &A_m, Matrix &L_m, double alpha = 1.0){
    
    
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
            //todo: verificar acesso de dados via Host
            //todo: modificar sycl::range - decl. unica
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



//Cholesky parallel 
bool cholesky_v2(sycl::queue &q, Matrix &A_m, Matrix &L_m, double alpha = 1.0){
    
    
    //check if there's a square Matrix declared: 
    // if (!A_m.square_m() || !L_m.square_m() || !A_m.is_symmetric())
    //{
    //    std::cout << "\nMatrix not symmetric\n";
    //    return 0;
    //} 
    // else ....
    
    //K represents iteration through this algorithm
    int *k = malloc_shared<int>(1, q);
    int *nr = malloc_shared<int>(1, q);
    nr[0] = A_m.n_row;
    
    //Set 2D buffers to host in devices
    sycl::range<2> num_items{static_cast<size_t>(nr[0]), static_cast<size_t>(nr[0])};
    
    // sycl::buffer<double,2> A_buf = {A_m.elem.data(), num_items};
    //Create a buffer with A data.
    sycl::buffer<double,2> L_buf = {A_m.elem.data(), num_items};


    // buffer<int, 1> k(data, range<1>(nElems));

        
    //Algoritmo de Cholesky Incompleto, iterado ao longo de sua coluna
    for(k[0] = 0; k[0] < nr[0]; k[0]++)
    {        
        //Range for division:
        sycl::range<2>c_div{static_cast<unsigned long>(nr[0]-(k[0]+1)),1};
        
        //sqrt in  l_kk of its diagonal entry
        auto eA = q.submit([&] (handler &h){
            /*define parameters to send device*/
            sycl::accessor L_acc{L_buf, h};
            // sycl::accessor A_acc{A_buf, h};
            int itr = k[0];
            h.single_task([=](){L_acc[itr][itr] = sqrt(L_acc[itr][itr]);});
        });
        
                
        //task B: calcular a coluna (a(i,j) /=a(j,j)
        //for( i = k+1; i < n; i++)
        auto eB = q.submit([&](handler &h) {
            h.depends_on(eA);
            //declare accessor to buffers
            sycl::accessor L_acc{L_buf, h};
            //todo: verificar acesso de dados via Host
            //todo: modificar sycl::range - decl. unica
            int p = k[0];
            h.parallel_for(c_div, [=](auto idx){
                int i = idx[0]+p+1;
                int j = idx[1]+p;
                L_acc[i][j] /= L_acc[p][p];                
            });            
        });
        // eB.wait();
        
        //task C:Elimination B(i,j) -= B(i,k) * B.(j,k)
    
            for(int j = k[0]+1; j < nr[0]; j++)
            {
                //Range for permutation:
            sycl::range<2> c_mod{static_cast<unsigned long>(nr[0]-j),1};
                //for(int i = j_0; i < n; i++)
                auto eC = q.submit([&](handler &h) {
                    h.depends_on({eB});
                        
                    //declare accessor to buffers
                    sycl::accessor L_acc{L_buf, h};

                    int p = k[0];
                    int j_0 = j; 
                    
                    h.parallel_for(c_mod, [=](auto idx){     
                        int i = j_0 + idx[0];
                        
                        
                        L_acc[i][j_0] -= L_acc[i][p] * L_acc[j_0][p] ;

                    });            

                });
                eC.wait();
              
                //print_m(L_m);
            }       
             
    }
    
    return true; 
}


#endif// LIBMAT_PARALLEL_HPP