###Hieu Pham

### AMS595: Fundamentals of Computing

###Group Project: Parallel Matrix multiplication



#### Introduction

The program is implement the Fox and Ring method to emulate matrix multiplication in a parallel computing settings, using `C++` and `MPI`

Some prerequisites: given 2 matrices $A$ and $B$

- Both number of processes and the matrix dimensions are of the power of 2
  - We will be using $p=2^n$ processes $(n > 2)$
  - Both matrices must be of $N\times N$ dimension ($N = 2^m, m > n $)
- For simplicity of this program, we will randomize each element of both matrices such that $A_{ij}, B_{ij} \in (-1,1]$



#### Running the program

The program can be run on Seawulf supercomputer

- after loading the slurm module, the job script can be submitted by typing`make` into the terminal
- `Ring()` and `Fox()` are called from `mat_mul.cpp` and should be run individually for execution time record 
  - they are commented out by default

- Any generated matrix/array can be printed using the `debug()` function
- results are stored in `output.log`



#### Methods

Given matrix $A=\begin{bmatrix}a_{11}&a_{12}&..&a_{1n}\\...\\a_{n1}&a_{n2}&..&a_{nn}\end{bmatrix}$ and $B=\begin{bmatrix}b_{11}&b_{12}&..&b_{1n}\\...\\b_{n1}&b_{n2}&..&b_{nn}\end{bmatrix}$

Let $C=A\cdot B = \begin{bmatrix}c_{11}&c_{12}&..&c_{1n}\\...\\c_{n1}&c_{n2}&..&c_{nn}\end{bmatrix}$



##### Ring method 

function: `void ring(const Matrix &A, const Matrix &B, Matrix &C);`

Given $p$ processes

At the $1^{st}$ step:

- For each process $p_i$

  - `ring()` will have local vectors `localRowA` and `localColB` , in which 

    - `localRowA` = $\begin{bmatrix}a_{i1}&a_{i2}&..&a_{in}\end{bmatrix}$ , taking the $i^{th}$ row of $A$
    - `localColB` = $\begin{bmatrix}b_{1i}\\b_{2i}\\..\\a_{ni}\end{bmatrix}$ ,  taking the $i^{th}$ column of $B$, which will be represented by a transposed vector in the  program

    

  -  $c_{ii} =$ `localRowA` $\cdot$ `localColB`  , in which $i = 1,2,...p$

- Once all processes are done

  - $C$ = $\begin{bmatrix}c_{11}&0&0&...&0&...\\
    0&c_{22}&0&...&0&...\\...\\0&0&0&...&c_{pp}&...\\...\end{bmatrix}$ 
  -  $C$ can collated from all processes's resulting matrix using `MPI_Allreduce()`

For the next $k$ steps $(k<n)$:

- For each process $p_i$, 
  - `localRowA`  will row down and take  the $(i+k)^{th}$ row of $A$ 
  - `localColB` will retain the $i^{th}$ column of $B$, 
  -  $c_{(i+k)i} =$ `localRowA` $\cdot$ `localColB`
    - if $i+k>n$ and $k < n-1$, we can make it so that `localRowA`  will roll back to $1^{st}$ row instead of going out of bound
- For illustration, at the $2^{nd}$ step ($k=1$):
  - $C$ = $\begin{bmatrix}c_{11}&0&0&...&0&...\\
    c_{12}&c_{22}&0&...&0&...\\0&c_{32}&c_{33}&...&0&...\\...\\0&0&0&...&c_{pp}&...\\0&0&0&...&c_{(p+1)p}&...\\...\end{bmatrix}$ 

- Each process is done when it has finished taking $n$ steps in total ($k=n-1$)
  - The resulting matrix $C$ after $n$ steps : 
    - $C$ = $\begin{bmatrix}c_{11}&c_{12}&c_{13}&...&c_{1p}&...\\
      c_{12}&c_{22}&c_{23}&...&c_{2p}&...\\...\\c_{p1}&c_{p2}&c_{p3}&...&c_{pp}&...\\...\\c_{n1}&c_{n2}&c_{n3}&...&c_{np}&...\end{bmatrix}$ 



Let call the $n$-step process above **Q**

If $p < n$ (which is a given in most situations), we will repeat **Q** for $\frac{n}{p}$ times

For each iteration $j$ ($j = 0,1,..,\frac{n}{p} -1$)

- `localColB` will take the $(i+j\times p)^{th}$ column of $B$, 
- → **Q** will populate $C$ from the  $(jp+1)^{th}$ column to  $((j+1)p)^{th}$ column 

$C$ will be fully populated once the loop is complete



##### Fox (BMR) method 

function: `void fox(const Matrix &A, const Matrix &B, Matrix &C);`

$A$ and $B$ will be flatten into one-dimentional vector `flatA` and `flatB` to reduce communication time between processes

- `flatA` = $\begin{bmatrix}a_{11}&a_{12}&...&a_{1n}&a_{21}&...&a_{nn}\end{bmatrix}$ 
- `flatB` = $\begin{bmatrix}b_{11}&b_{12}&...&b_{1n}&b_{21}&...&b_{nn}\end{bmatrix}$ 

The BMR method divides each matrix into blocks assigned to each process. Each process:

- Holds a block of $A$ (called `localA`) and a block of $B$(called `localB`).
- Part of the product matrix $C$ `localC` will be computed based on those blocks

Given $p$ processes

For each process $p_i$:

- **Broadcast**:  receives a row block of  $A$ and $B$, broadcasted by the root $p_1$

  - Each process will handle a block of rows of matrix $A$ and a block of rows of matrix $B$ . `blockSize` is computed based on the number of rows each process will handle (`n / size`).

  - Using `MPI_Scatter`, each process receives a portion of $A$ and $B$ to work on.

- **Multiply**:  $p_i$ performs matrix multiplication between the broadcasted row block from `A` and its own `B` block.
  - The computation result is stored in `localC`, which accumulates partial results for each process.

- **Roll**: $p_i$ sends its `localB` block to  $p_{i+1} $ and receives the `localB` block from $p_{i+1}$.
  - The `MPI_Sendrecv_replace` function is used to "roll" the blocks of `B` in a circular fashion among the processes.

Then the abovementioned process will repeat until each process has multiplied its `A` block with every row block of `B` $p$ times



#### Result

We test the matrix multiplication given these specifications

- $p=2^2, 2^4, 2^6$
- $N=2^8, 2^{10}, 2^{12}$

Belows are the results after running the program

- **Sequential**

  - For $2^8$ dimensions, time elapsed during the job: 0.03716s.
  
  
    - For $2^{10}$ dimensions, time elapsed during the job: 3.40246s.
  
  
    - For $2^{12}$ dimensions, time elapsed during the job: 795.83825s.
  


- **Fox Method**

  - For $2^2$ processes and $2^8$ dimensions, time elapsed during the job: 0.00529s.

  - For $2^2$ processes and $2^{10}$ dimensions, time elapsed during the job: 0.33534s.

  -  For $2^2$ processes and $2^{12}$ dimensions, time elapsed during the job: 30.33475s.

  - For $2^4$ processes and $2^8$ dimensions, time elapsed during the job: 0.00317s.

  - For $2^4$ processes and $2^{10}$ dimensions, time elapsed during the job: 0.08171s.

  - For $2^4$ processes and $2^{12}$ dimensions, time elapsed during the job: 8.02823s.

  - For $2^6$ processes and $2^8$ dimensions, time elapsed during the job: 0.00327s.

  - For $2^6$ processes and $2^{10}$ dimensions, time elapsed during the job: 0.03893s.

  - For $2^6$ processes and $2^{12}$ dimensions, time elapsed during the job: 1.69635s.


- **Ring Method**

  - For $2^2$ processes and $2^8$ dimensions, time elapsed during the job: 0.01618s.

  - For $2^2$ processes and $2^{10}$ dimensions, time elapsed during the job: 1.03047s.

  - For $2^2$ processes and $2^{12}$ dimensions, time elapsed during the job: 89.33743s

  - For $2^4$ processes and $2^8$ dimensions, time elapsed during the job: 0.02655s.

  - For $2^4$ processes and $2^{10}$ dimensions, time elapsed during the job: 0.88076s.

  - For $2^4$ processes and $2^{12}$ dimensions, time elapsed during the job: 72.43762s.

  - For $2^6$ processes and $2^8$ dimensions, time elapsed during the job: 0.04764s.

  - For $2^6$ processes and $2^{10}$ dimensions, time elapsed during the job: 0.84287s.

  - For $2^6$ processes and $2^{12}$ dimensions, time elapsed during the job: 59.85964s.




#### Analysis

The speedup is calculated as:

​		 $\text{Speedup} = \frac{\text{Sequential Time}}{\text{Parallel time}} $

Below are the speedup curve for both methods:

#####![Screenshot 2024-12-11 at 1.01.54 AM](/Users/phamquoctrung/Library/Application Support/typora-user-images/Screenshot 2024-12-11 at 1.01.54 AM.png)

The speedup curves for matrix multiplication highlight the benefits of parallelization over sequential execution. 

- Sequential processing time increases exponentially with matrix size, making parallel approaches essential for large matrices.

**Fox Method** demonstrates superior scalability and efficiency, achieving significant speedups, especially for larger dimensions (e.g., $2^{12}$). It effectively balances computation and communication, making it ideal for dense matrix multiplication with high process counts. 

Conversely, **Ring Method** exhibits slower speedups due to higher communication overhead, particularly with smaller matrices or higher process counts. While it performs reasonably well for large matrices, it lags behind the Fox Method in efficiency.

In summary, the Fox Method is better suited for matrix multiplication, especially for large matrices and high process counts, while the Ring Method is less efficient due to its communication-heavy design. For smaller matrices, both methods offer limited speedup due to communication overhead.