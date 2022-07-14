#include <vector>


#ifndef STRUCTS_H
#define STRUCTS_H

typedef std::vector<double> DVector;
typedef std::vector<int> IVector;

class Matrix
{
    // private:
public:
    long int n_row, n_col;
    DVector elem;

    // long int col() const {{return n_col;}}
    // long int row() const {{return n_row;}}
    // Returns size matrix
    long int size() { return elem.size(); }

    // Constructor
    Matrix(long int col, long int row)
    {
        n_col = col;
        n_row = row;
        elem.resize(n_col * n_row);
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
    double at(int a, int b) { return elem[n_col * b + a]; }

    bool eye(double v = 1.0)
    {
        std::memset(elem.data(), 0, sizeof(elem));
        for (long int i = 0; i < n_row; i++)
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
        for (long int i = 0; i < n_row; i++)
        {
            for (long int j = 0; j < n_col; j++)
            {
                if (elem[i * n_col + j] != elem[j * n_col + i])
                    return false;
            }
        }
        return true;
    }
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
}wm;
// Discrete space state
typedef struct ss_d {
        Matrix *x;
        Matrix *xp;
        Matrix *z;
        Matrix *Phi;
        Matrix *G;
        Matrix *H;
        struct wm wm;
}ss_d;

#endif // STRUCTS_H