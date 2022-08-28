#include <vector>
#include <iostream>
#include <cstring>

#ifndef STRUCTS_H
#define STRUCTS_H

typedef std::vector<double> DVector;
typedef std::vector<int> IVector;


// todo:create relation between Matrix & Vector
class Matrix
{
    // private:
public:
    int n_row, n_col;
    DVector elem;

    // int col() const {{return n_col;}}
    // int row() const {{return n_row;}}
    // Returns size matrix
    int size() { return elem.size(); }

    // Constructor
    Matrix(int row, int col)
    {
        n_col = col;
        n_row = row;
        elem.resize(n_col * n_row, 0.00);
    }

    // Destructor
    ~Matrix()
    {
        elem.clear();
    }

    friend bool operator!=(const Matrix &A, const Matrix &B)
    {
        return ((A.n_row != B.n_row || A.n_col != B.n_row) || A.elem != B.elem);
    }
    bool is_symmetric() 
    {
        for(int i = 0; i < n_row; i++)
        {
            for(int j = 0; i< n_col; j++)
            {
                if(elem[n_row*i+j] != elem[n_row*j+i]) return false;
            }
        }
        return true;
    }
    bool square_m()
    {
        return n_row == n_col;
    }
    double & at(int i, int j){ return (this)->elem[i*n_col+j];}
    const double & at(int i, int j) const { return (this)->elem[i*n_col+j];}
    //polimorphism ?
    // double operator[](int b) { return elem[b]; }
    bool eye(double v = 1.0)
    {
        std::memset(elem.data(), 0, sizeof(elem));
        for (int i = 0; i < n_row; i++)
        {
            elem[n_col * i + i] = v;
        }
        return true;
    }

    void zeros(){
        std::memset(elem.data(), 0, sizeof(elem));
        
    }
    bool isSymmetric()
    {
        // is it a square matrix?
        if (n_col != n_row)
            return 0;

        // else... is Symmetric?
        for (int i = 0; i < n_row; i++)
        {
            for (int j = 0; j < n_col; j++)
            {
                if (elem[i * n_col + j] != elem[j * n_col + i])
                    return false;
            }
        }
        return true;
    }

    double *data() {return elem.data();}

    void copy_of(Matrix &A)
    {
        n_row = A.n_row;
        n_col = A.n_col;

        this->elem.clear();
        elem = A.elem;
    }
    bool equal_dim(Matrix &B){
        return ((n_row == B.n_row) && (n_col == B.n_col));}
};



//################################## IMU's structs #############################
struct axes {
        double x;
        double y;
        double z;
};
// Data from IMU
struct IMU{
        struct axes acel;
        struct axes rate;
        struct axes mag;
};

struct data_IMU {
        struct IMU bin;
        struct IMU scaled;
        struct IMU adj;
};

struct euler {
        double yaw;
        double pitch;
        double roll;
};

// ############################### State Space's structs ###############################
// Weight matrices
typedef struct wm {
        Matrix *Q;
        Matrix *Qeta;
        Matrix *Qeta_sr;
        Matrix *R;
        Matrix *R_sr;
        Matrix *P;
        Matrix *P_sr;
        Matrix *Pp;

        Matrix *Pz;
        Matrix *Pxz;
        DVector *W;
        double kappa;

}wm;
// Discrete space state
typedef struct ss_d {
        Matrix *x;
        Matrix *z;
        Matrix *xp;
        Matrix *Phi;
        Matrix *G;
        Matrix *H;

        double dt;
        Matrix *Xi;
        Matrix *fXi;
        Matrix *hXi;
        Matrix *zp; 
        Matrix *K;
        
        struct wm wm;
}ss_d;

#endif // STRUCTS_H