
#include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif


#include "libmat.hpp"

using namespace sycl;
using namespace std;

constexpr int size_m = 3;

int main() {
     
    
    Matrix A(size_m, size_m);
    
    A.elem = { 4, 12, -16,
              12, 37, -43,
             -16,-43, 98};
    
    Matrix B(size_m, size_m);
    B.eye(0);
    Matrix C1(size_m, size_m);
    Matrix C2(size_m, size_m);
    //C1.eye(0); 
    //C2.eye(0);
    
    bool exec = false; 
    double alpha = 5.27;
    //print_m(A);
    //print_m(B);
    
    try {
        
        sycl::device device = sycl::device(sycl::default_selector());
        sycl::queue q;
        std::vector<sycl::event> event_list;
        // Print out the device information used for the kernel code.
        std::cout << "Running on device: "
                  << q.get_device().get_info<info::device::name>() << "\n";
        std::cout << "Vector size: " << A.size() << "\n";

        
        // Vector addition in DPC++
        //exec = sum_m(q, A, B, C1, 2.0, 5.0);  
        //exec = times_m(q, A,B,C1);
        //exec = timesc_m(q, alpha, B, C1);
        exec = cholesky_v1(q,A, C1);
        
    } catch (sycl::exception const &e)
                        {
        std::cout << "An exception is caught for vector add.\n";
        std::terminate();
    }    
    
    //sum_m(A, B, C2, 2.0, 5.0);
    //times_m(A,B,C2);
    //timesc_m(alpha, B, C2);
    cholesky(A, C2);
    
    if(exec == true) 
    {
        std::cout <<"executed\n";
        
    if(C1.elem != C2.elem)
    {
        std::cout << "Error on sum_m operation !\n";       
    }
    else std::cout << "Calc complete ! Matrices are equal\n"; 
                     
                     
    }
    
    std::cout << "Paralel\n";
    print_m(C1);
    std::cout << "seq\n";
    print_m(C2);   
    
    return 0;   
}
