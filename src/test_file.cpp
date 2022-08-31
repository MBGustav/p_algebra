#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
// #include "oneapi/mkl.hpp"
#include "libmat.hpp"

#include "libmat_parallel.hpp"

#include <algorithm> 




using namespace sycl;
//using namespace std;


// void chol_mkl(sycl::queue &queue, double *L, double *U, int64_t nb,double alpha){
// /* # ====================
    
//     #U    -> matriz Input
//     #nb   -> numero de linhas do bloco U
//     #alpha-> multiplicador escalar na função de Cholesky
    
//    # ====================*/
    
//     // # Make a copy of P (P must remain unchanged)/
//     //Trabalhar como retornar para host_local ? 
//     //double *U = sycl::malloc_shared<double>(nb*nb, queue);
//     for(int i = 0; i < nb*nb; i++)
//         U[i] = 0;
//     //zero(U,nb,nb);
    
//     auto copy_L = blas::row_major::copy(queue, nb*nb, 
//                                         L, 1, U, 1);
//     copy_L.wait();
    
//     std::int64_t scratchpad_size = lapack::potrf_scratchpad_size<double>(queue, lowerM, nb, nb);
//     double *scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
    
//     auto event1 = blas::scal(queue, nb*nb, alpha, U, 1.0);
//     event1.wait_and_throw();
    
//     auto event2 = lapack::potrf(queue, lowerM, nb, U, nb, scratchpad, scratchpad_size );
//     event2.wait_and_throw();
    
//     for(int i = 0; i<nb; i++){
//         for(int j = 0; j<nb;j++){
//             if(i>j) U[i*nb+j] = 0;
//         }
//     }
    
//     free(scratchpad, queue);
// }

constexpr int size_m = 1000;

int main() {
     
    
    Matrix A(size_m, size_m);
    
    get_inversible_matrix(A, 20);
    
    Matrix B(size_m, size_m);
    B.eye(1);
    

    Matrix C_dpc(size_m, size_m);
    Matrix C_seq(size_m, size_m);
    
    
    bool exec = false; 
    double alpha = 5.27;
    double par_chol = 0; 
    double seq_chol = 0; 
    double mkl_chol = 0; 

    
    try {
        
        sycl::device device = sycl::device(sycl::default_selector());
        sycl::queue q{property::queue::enable_profiling{}} ;
        std::vector<sycl::event> event_list;
        // Print out the device information used for the kernel code.
        std::cout << "Running on device: "
                  << q.get_device().get_info<info::device::name>() << "\n";
        std::cout << "Vector size: " << A.size() << "\n";

        
        // Vector addition in DPC++
        //exec = sum_m(q, A, B, C_dpc, 2.0, 5.0);  
        //exec = times_m(q, A,B,C_dpc);
        //exec = timesc_m(q, alpha, B, C_dpc);
        auto beg = std::chrono::high_resolution_clock::now();
        exec = cholesky_v2(q, A, C_dpc);
        auto end = std::chrono::high_resolution_clock::now();
        par_chol = std::chrono::duration_cast<std::chrono::duration<double>>(end - beg).count();
        
 //      beg = std::chrono::high_resolution_clock::now();
 //       chol_mkl(q, A.elem.data(),C_mkl.elem.data(), size_m, 1.0);
  //      end = std::chrono::high_resolution_clock::now();
   //     mkl_chol = std::chrono::duration_cast<std::chrono::duration<double>>(end - beg).count();
        
        //std::cout << "Paralel Time: " << par_chol.count() << "\n";
        
    } catch (sycl::exception const &e)                    
    {
        std::cout << "An exception is caught for vector add.\n";
        std::terminate();
    }
    
    //sum_m(A, B, C_seq, 2.0, 5.0);
    //times_m(A,B,C_seq);
    //timesc_m(alpha, B, C_seq);
    
    auto beg = std::chrono::high_resolution_clock::now();
    cholesky(&A, &C_seq);
    auto end = std::chrono::high_resolution_clock::now();
        seq_chol = std::chrono::duration_cast<std::chrono::duration<double>>(end - beg).count();
    
        
    if(exec == true) 
    {std::cout <<"executed\n";
        
    if(C_dpc.elem != C_seq.elem)
    {
        std::cout << "Error on operation !\n";       
    }
    else std::cout << "Calc complete ! Matrices are equal\n"; 
                     
                     
    
    
    std::cout << "DPC++ Time: " << par_chol *1e-9 << " sec\n";
    //print_m(C_dpc);
    std::cout << "Sequential Time: " << seq_chol *1e-9<<" sec\n";
    //print_m(C_seq);   
    (par_chol < seq_chol) ? std::cout << "paralelo ": std::cout << "sequencial ";
    std::cout << "wins! :D \t Time comparation: " << std::max(par_chol, seq_chol)/std::min(par_chol, seq_chol) <<"\n";
    }
    return 0;   
}
