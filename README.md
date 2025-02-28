### Hieu Pham

### AMS595: Fundamentals of Computing

### Group Project: Parallel Matrix Multiplication

#### Introduction

The program implements the Fox and Ring methods to emulate matrix multiplication in a parallel computing setting, using `C++` and `MPI`.

Some prerequisites: given two matrices $A$ and $B$:

- Both the number of processes and the matrix dimensions are powers of 2.
  - We will be using $p=2^n$ processes $(n > 2)$.
  - Both matrices must be of $N\times N$ dimension ($N = 2^m, m > n$).
- For simplicity, we will randomize each element of both matrices such that $A_{ij}, B_{ij} \in (-1,1]$.

#### Running the Program

The program can be run on the Seawulf supercomputer:

- After loading the Slurm module, the job script can be submitted by typing `make` into the terminal.
- `Ring()` and `Fox()` are called from `mat_mul.cpp` and should be run individually for execution time records. They are commented out by default.
- Any generated matrix/array can be printed using the `debug()` function.
- Results are stored in `output.log`.

#### Methods

Given matrices:

```math
A = \begin{bmatrix}a_{11}&a_{12}&..&a_{1n}\\ ... \\ a_{n1}&a_{n2}&..&a_{nn}\end{bmatrix}
```

and

```math
B = \begin{bmatrix}b_{11}&b_{12}&..&b_{1n}\\ ... \\ b_{n1}&b_{n2}&..&b_{nn}\end{bmatrix}
```

Let

```math
C = A \cdot B = \begin{bmatrix}c_{11}&c_{12}&..&c_{1n}\\ ... \\ c_{n1}&c_{n2}&..&c_{nn}\end{bmatrix}
```

##### Ring Method

Function:

```cpp
void ring(const Matrix &A, const Matrix &B, Matrix &C);
```

Given $p$ processes:

At the first step:

- For each process $p_i$:
  - `ring()` will have local vectors `localRowA` and `localColB`, where:
    - `localRowA` = `[a_{i1}, a_{i2}, ..., a_{in}]`, taking the $i^{th}$ row of $A$.
    - `localColB` = `[b_{1i}, b_{2i}, ..., b_{ni}]^T`, taking the $i^{th}$ column of $B$ (represented as a transposed vector).
  - $c_{ii} =$ `localRowA` $\cdot$ `localColB`, where $i = 1,2,...p$.

Once all processes are done:

```math
C = \begin{bmatrix}c_{11}&0&0&...&0&...\\ 0&c_{22}&0&...&0&...\\ ...\\ 0&0&0&...&c_{pp}&...\\ ...\end{bmatrix}
```

$C$ can be collated from all processes using `MPI_Allreduce()`.

For the next $k$ steps $(k<n)$:

- Each process $p_i$:
  - `localRowA` shifts down to the $(i+k)^{th}$ row of $A$.
  - `localColB` retains the $i^{th}$ column of $B$.
  - $c_{(i+k)i} =$ `localRowA` $\cdot$ `localColB$.

If $i+k>n$ and $k < n-1$, `localRowA` rolls back to the first row instead of going out of bounds.

The process completes when each process has taken $n$ steps in total ($k=n-1$).

##### Fox (BMR) Method

Function:

```cpp
void fox(const Matrix &A, const Matrix &B, Matrix &C);
```

$A$ and $B$ are flattened into one-dimensional vectors `flatA` and `flatB` to reduce communication time:

```math
flatA = [a_{11}, a_{12}, ..., a_{1n}, a_{21}, ..., a_{nn}]
```

```math
flatB = [b_{11}, b_{12}, ..., b_{1n}, b_{21}, ..., b_{nn}]
```

The BMR method divides each matrix into blocks assigned to each process. Each process:

- Holds a block of $A$ (`localA`) and a block of $B` (`localB`).
- Computes a part of the product matrix `C` (`localC`).

Each process performs:

1. **Broadcast:** Receives a row block of $A$ and $B$.
2. **Multiply:** Computes the multiplication of the received blocks.
3. **Roll:** Sends its `localB` block to $p_{i+1}$ and receives `localB` from $p_{i-1}$.
4. Repeat until every process has multiplied all relevant blocks.

#### Results

We test matrix multiplication under these specifications:

- $p=2^2, 2^4, 2^6$
- $N=2^8, 2^{10}, 2^{12}$

Results:

**Sequential:**

- $2^8$: 0.03716s
- $2^{10}$: 3.40246s
- $2^{12}$: 795.83825s

**Fox Method:**

- $2^2, 2^8$: 0.00529s
- $2^2, 2^{10}$: 0.33534s
- $2^2, 2^{12}$: 30.33475s
- $2^4, 2^8$: 0.00317s
- $2^4, 2^{10}$: 0.08171s
- $2^4, 2^{12}$: 8.02823s
- $2^6, 2^8$: 0.00327s
- $2^6, 2^{10}$: 0.03893s
- $2^6, 2^{12}$: 1.69635s

**Ring Method:**

- $2^2, 2^8$: 0.01618s
- $2^2, 2^{10}$: 1.03047s
- $2^2, 2^{12}$: 89.33743s
- $2^4, 2^8$: 0.02655s
- $2^4, 2^{10}$: 0.88076s
- $2^4, 2^{12}$: 72.43762s
- $2^6, 2^8$: 0.04764s
- $2^6, 2^{10}$: 0.84287s
- $2^6, 2^{12}$: 59.85964s
