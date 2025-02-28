#include <cstdlib>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <functional>


#define LINE 50

using Matrix = std::vector<std::vector<float> >;

void debug(const Matrix &v){
    int current, total;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &current);
    if(current > 0){

        int dummy{0};
		MPI_Recv(&dummy, 1, MPI_INT, current-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    std::cout << "Process " << current << std::endl;
    for (int i = 0; i < v.size(); i++) {
        for(int j = 0; j < v[i].size(); j++){
            std::cout << v[i][j] << "\t";  // Example data for each process
        }
        std::cout << std::endl;
    }
    std::cout << std::endl <<std::string(LINE,'-') << std::endl;

	if (current < total-1) MPI_Send(&current, 1, MPI_INT, current+1, 0, MPI_COMM_WORLD);
}
void debug(const std::vector<float> &v){
    int current, total;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &current);
    if(current > 0){

        int dummy{0};
		MPI_Recv(&dummy, 1, MPI_FLOAT, current-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    std::cout << "Process " << current << std::endl;
    for (int i = 0; i < v.size(); ++i) {
        std:: cout << v[i] << "\t";  // Example data for each process
    }
    std::cout << std::endl <<std::string(LINE,'-') << std::endl;

	if (current < total-1) MPI_Send(&current, 1, MPI_FLOAT, current+1, 0, MPI_COMM_WORLD);
}

void randomize(std::vector<float> &v){
    // Create a random number generator
    std::random_device rng;
    std::mt19937 engine{rng()};  // Generates random integers
    std::uniform_real_distribution<float> dist {-1, 1};
    auto gen = [&](){return dist(engine);};
    std::generate(v.begin(), v.end(), gen);
    return;
}

void randomizeMatrix(Matrix &a){
    for(int i = 0; i < a.size();i++){
        randomize(a[i]);
        //a[i][i] = 1;
        //for(int j = 0; j < a.size();j++) a[i][j] = i*a.size()+j+1;
    }
    return;
}

Matrix transpose(const Matrix &a){
    int n = a.size();
    Matrix ret(n, std::vector<float>(n,0));
    for(int i = 0;i < n;i++){
        for(int j = 0;j < n;j++){
            ret[i][j] = a[j][i];
        }
    }
    return ret;
}

// Function to print a matrix
void printMatrix(const Matrix &matrix) {
    for (const auto &row : matrix) {
        for (float val : row)
            std::cout << val << "\t";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
float dotProd(const std::vector<float> &a, const std::vector<float> &b){
    float ret{0};
    for(int i = 0; i < a.size();i++){
        ret += a[i] * b[i];
    }
    return ret;
}


// Helper function to flatten a 2D vector into a 1D vector for MPI communication
std::vector<float> flatten(const Matrix& matrix) {
    int r = matrix.size();
    int c = matrix[0].size();
    std::vector<float> flat(r * c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            flat[i * c + j] = matrix[i][j];
        }
    }
    return flat;
}

// Helper function to unflatten a 1D vector into a 2D vector after MPI communication
void unflatten(const std::vector<float>& flat, Matrix& matrix) {
    int r = matrix.size();
    int c = matrix[0].size();
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            matrix[i][j] = flat[i * c + j];
        }
    }
}
