#include <iterator>
#include <mpi.h>
#include "utils.cpp"


// Ring Method for Matrix Multiplication
void ring(const Matrix &A, const Matrix &B, Matrix &C) {
    int current, total;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &current);
    int n = A.size();
    std::vector<float> localRowA(n,0);
    std::vector<float> localColB(n,0);
    Matrix T = transpose(B);
    Matrix tempC(n, std::vector<float>(n,0));

    for(int step = 0; step < n; step += total){
        // Column
        if (current == 0) {
            std::copy(T[step].begin(), T[step].end(), localColB.begin());
            for(int i = 1; i < total;i++){
                int idx = (i+step)%n;
                MPI_Send(&T[idx][0], n, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
            }
        }
        else{
            MPI_Recv(&localColB[0], n, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // Row
        for (int i = 0; i < n;i++){
            if (current == 0) {
                std::copy(A[i].begin(), A[i].end(), localRowA.begin());
                for(int j = 1; j < total;j++){
                    int idx = (i+j)%n;
                    MPI_Send(&A[idx][0], n, MPI_FLOAT, j, 0, MPI_COMM_WORLD);
                }
            }
            else{
                MPI_Recv(&localRowA[0], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            int r = (i+current)%n;
            int c = (step+current)%n;
            tempC[r][c] = dotProd(localRowA, localColB);
        }
    }
    for(int i = 0; i < n;i++){
        MPI_Allreduce(&tempC[i][0],&C[i][0], n, MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD);
    }
    // debug(C);
}

void fox(const Matrix &A, const Matrix &B, Matrix &C) {
    int current, total;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &current);
    int n = A.size();

    int blockSize = n / total;  // Rows handled by each process

     // Local blocks of matrices
     Matrix localA(blockSize, std::vector<float>(n));
     Matrix localB(blockSize, std::vector<float>(n));
     Matrix localC(blockSize, std::vector<float>(n, 0)); // Initialize to zero

     // Flatten matrices A and B for MPI scatter
     std::vector<float> flatA, flatB;
     if (current == 0) {
         flatA = flatten(A);
         flatB = flatten(B);
     }

     // Buffers for scattering and gathering
     std::vector<float> flatLocalA(blockSize * n);
     std::vector<float> flatLocalB(blockSize * n);

     // Scatter rows of A and B among the processes
     MPI_Scatter(flatA.data(), blockSize * n, MPI_FLOAT, flatLocalA.data(), blockSize * n, MPI_FLOAT, 0, MPI_COMM_WORLD);
     MPI_Scatter(flatB.data(), blockSize * n, MPI_FLOAT, flatLocalB.data(), blockSize * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

     // Unflatten the received blocks into 2D vectors
     unflatten(flatLocalA, localA);
     unflatten(flatLocalB, localB);

     // Temporary buffer for ring communication of B
     Matrix tempB = localB;

     int currentBlockRow = current; // Track the current row block of B
     for (int step = 0; step < total; ++step) {
         // Multiply localA block with the rotated localB block
         for (int i = 0; i < blockSize; ++i) {
             for (int j = 0; j < n; ++j) {
                 for (int k = 0; k < blockSize; ++k) {
                     localC[i][j] += localA[i][k + (currentBlockRow * blockSize)] * localB[k][j];
                 }
             }
         }

         // Flatten localB for the ring communication
         std::vector<float> flatLocalB = flatten(localB);

         // Roll (shift) the matrix B block to the next process
         int next = (current + 1) % total;
         int prev = (current - 1 + total) % total;
         MPI_Sendrecv_replace(flatLocalB.data(), blockSize * n, MPI_FLOAT, next, 0, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

         // Unflatten the received data back to localB
         unflatten(flatLocalB, localB);

         // Update current block row index for the next roll
         currentBlockRow = (currentBlockRow - 1 + total) % total;
     }

     // Flatten localC for gathering
     std::vector<float> flatLocalC = flatten(localC);
     std::vector<float> flatC(n * n);
     MPI_Allgather(flatLocalC.data(), blockSize * n, MPI_FLOAT, flatC.data(), blockSize * n, MPI_FLOAT, MPI_COMM_WORLD);

     // Unflatten the final gathered result into the matrix C at root process
     unflatten(flatC, C);
     //debug(C);
}

int main (int argc, char** argv){
    int current, total;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &current);

    double start{0.0}, end{0.0};
    // Initialize the array for each process
    uint64_t n = static_cast<uint64_t>(std::stof(argv[1]));
    Matrix A(n, std::vector<float>(n));
    Matrix B(n, std::vector<float>(n));
    Matrix C(n, std::vector<float>(n,0));

    if(current == 0){
        randomizeMatrix(A);
        randomizeMatrix(B);
        //printMatrix(A);
        start = MPI_Wtime();
    }
    //ring(A, B, C);
    fox(A, B, C);
    //debug(C);

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    if (current ==0) printf("For %d processes and %d dimensions, time elapsed during the job: %.5fs.\n", total, n, end - start);
    MPI_Finalize();
    return 0;
}
