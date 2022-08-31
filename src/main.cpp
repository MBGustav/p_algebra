// #include <CL/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include <algorithm> 
#include <fstream>


#include "structs.hpp"
#include "libmat_parallel.hpp"
#include "libkf_parallel.cpp"
#include "time_counter.cpp"

// #include "oneapi/mkl.hpp"
// namespace blas = oneapi::mkl::blas;
// namespace lapack = oneapi::mkl::lapack;


using namespace sycl;


//Returns an array with (pseudo) random values from radar
double *GetRadar(int Nsamples, double dt, int seed);

//Seed to GetRadar
constexpr int seed = 777;

int main() 
{
    //create file : TestRadarUKF.csv

    std::ofstream myfile("TestRadarUKF_serial.csv");
    myfile << "time,Position,Velocity,Altitude\n";

    // Declare initial Structs
    wm *wm = new struct wm;
    ss_d *ss = new struct ss_d;

    //todo: devo considerar como uma variavel em ss_d ?
    int n = 3; 
    
    int nz = 1;
    double dt = 0.05;

    //Time Marker - it's time to benchmark !
    time_counter time((int) 20/dt);

    //Initial Matrices Declaration
    Matrix *x = new Matrix(n, 1);   // State vector
    Matrix *Q = new Matrix(n, n);   // Transition Cov. Matrix
    Matrix *R = new Matrix(nz, nz); // Measure Cov. Matrix
    Matrix *P = new Matrix(n, nz);  // Error Cov. Matrix
    Matrix *z = new Matrix(nz, nz);  // In this example z has 1 state -> Matrix (1, 1)

    P->eye(100);
    x->elem = {0.0, 90.0, 1100.0};
    R->at(0,0) = 100.0;

    wm->Q = Q;
    wm->R = R;
    wm->P = P;
    wm->kappa = 0.0;
    
    ss->x = x;
    ss->z = z;
    ss->dt = dt;
    
    int Nsamples = (int) 20/dt;
    double pos, alt, vel;
    double *Radius = GetRadar(Nsamples, dt, seed);
    DVector PosSaved(Nsamples);
    DVector AltSaved(Nsamples);
    DVector VelSaved(Nsamples);
    DVector YSaved(Nsamples);
    DVector t(Nsamples);

    DVector W(2*n+1);
    for(int i =0; i < W.size();i++) W[i] = i+1;
    Matrix  N(n, n);
    for(int i =0; i < N.size();i++) N.elem[i] = i+8;
    Matrix  S(2*n+1,n);
    for(int i =0; i < S.size();i++) S.elem[i] = i+5;
    
    Matrix xcov(n, n);
    Matrix xm(n, 1); 
    

    try {
         
        //About Accelerator Device & Queue
        sycl::device device = sycl::device(sycl::default_selector());
        std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
        sycl::queue queue(device, async_handler);
        std::vector<sycl::event> event_list;


        // for(int i = 0; i < Nsamples;i++){
        //     t[i] = dt*i; 
        //     z->elem[0] = Radius[i]; // Simulation - obtaining radar signal

        //     //Update and Store in Vectors
            
        //     time.add_marker();
        //     ukf_p(queue, wm, ss);
        //     time.end_marker("ukf_predict");


        //     time.add_marker();
        //     ukf_u(queue, wm, ss);
        //     time.end_marker("ukf_update");


        //     pos = x->elem[0];
        //     alt = x->elem[1];
        //     vel = x->elem[2];
            
        //     myfile << pos << ", " << vel << ", " << alt << "\n";
        // }
        print_m(xm);
        print_m(xcov);



    } catch (const exception &e) {
        std::cerr << "An exception occurred: "
                  << e.what() << std::endl;
        exit(1);

    }


    myfile.close();
    time.data2csv("UKF_BenchMark_Par.csv");
    return 0;   
}

double *GetRadar(int Nsamples,double dt, int seed){
	int posp = 0;
	int i = 0; 

	std::default_random_engine generator;
	generator.seed(seed);
	std::normal_distribution<double> distribution(0.0, 1.0);

	double *radar = new double[Nsamples];

	double vel, v, r, alt, pos;
	while (i < Nsamples){ 
        
		vel = 100  +  5*distribution(generator);
        alt = 1000 + 10*distribution(generator);
        pos = posp + vel*dt;

        v = pos *.05 * distribution(generator);
        r = sqrt(pos*pos + alt*alt) +v;
        radar[i++] = r;
        posp = pos;
	}

	return radar;
}



