#!/usr/bin/env bash

#SBATCH --job-name=ams530_pj3
#SBATCH --output=output.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=40
#SBATCH --time=180:00
#SBATCH -p short-40core

# load an MPI module
module load mvapich2/gcc13.2/2.3.7

# set env variables which may help performance

export MV2_HOMOGENEOUS_CLUSTER=1
export MV2_ENABLE_AFFINITY=0

# compile the code with the mpi compiler wrapper
## Question 2.1
mpic++ matrix_mul.cpp -o r1.exe


# execute the code with MPI
echo -e  "\n-----------------------Question 3.1---------------------------\n"
mpirun -np 4 ./r1.exe 4
