#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

/* Program Parameters */
int N = 6000;
#define MAXN 6000
/* Matrices */
float *A, *B;
volatile float AA[MAXN][MAXN], BB[MAXN][MAXN];


/* Initialize A and B*/
void initialize_inputs() {
    int row, col;
    
    srand((unsigned)time(NULL));
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[row*N+col] = (float)rand() / 32768.0;
            B[row*N+col] = 0.0;
            AA[row][col] = A[row*N+col];
            BB[row][col] = 0.0;
        }
    }
    
}


void print_output(){
    int r,c;
    printf("\nB =\n");
    for(r=0;r<N;r++){
        for(c=0;c<N;c++){
            printf("%5.5f%s", B[r*N+c], (c < N-1) ? ", " : ";\n");
        }
    }
}
void print_output2(){
    printf("\nBB =\n");
    int row, c;
    for(row=0;row<N;row++){
        for(c=0;c<N;c++){
             printf("%5.5f%s", BB[row][c], (c < N-1) ? ", " : ";\n");
        }
    }
}
void matrixNorm() {
    int row, col;
    float mu, sigma; // Mean and Standard Deviation
    
    for (col=0; col < N; col++) {
        mu = 0.0;
        for (row=0; row < N; row++)
            mu += AA[row][col];
        mu /= (float) N;
        sigma = 0.0;
        for (row=0; row < N; row++)
            sigma += powf(AA[row][col] - mu, 2.0);
        sigma /= (float) N;
        sigma = sqrt(sigma);
        for (row=0; row < N; row++) {
            if (sigma == 0.0)
                BB[row][col] = 0.0;
            else
                BB[row][col] = (AA[row][col] - mu) / sigma;
        }
    }
    
}


/* device function */


__global__ void matrixNorm(float *d_a, float *d_b, int n) {

    // get thread id(col) in the global scope
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("id:%d\n",col);
    int mu, sigma, row;
    mu = 0.0;
    // each threads load one element from global to shared mem.
    extern __shared__ float sdata[];
    
    for(row=0;row<n;row++){
        mu += d_a[row + col*n];
    }
    mu /= n;

    // make sure the calculation of mean is completed
    __syncthreads();

    sigma = 0.0;
    for (row=0; row < n; row++){
        sigma += powf(d_a[row+col*n] - mu, 2.0);
    }   
        sigma /= (float)n;
        sigma = sqrt(float(sigma));
    // make sure the calculation of standard deviation is completed
    __syncthreads();

    

    for(row=0;row<n;row++){
        if(sigma==0.0)
            d_b[row*n+col]=0.0;
        else
        d_b[row*n+col] = (d_a[row+col*n] - mu) / sigma;
    }

} 


int main(int argc, char **argv) {
    
    N = atoi(argv[1]);

    /* Initialize A and B */
    
    // allocate host memory for A, B
    A = (float*)malloc(4*N*N);
    B = (float*)malloc(4*N*N);

    
    initialize_inputs();

    // allocate device memory
    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, 4*N*N);
    cudaMalloc((void**)&d_B, 4*N*N);

    printf("\n--------------------- CUDA Start------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");

        /* Start Clock */

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // start to count execution time of GPU version
    cudaEventRecord(start,0);
    
    
    // copy the host data to device
    cudaMemcpy((void*)d_A, (void*)A, 4*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_B, (void*)B, 4*N*N, cudaMemcpyHostToDevice);
    
    // set up dimension of grid and block, 2-dim gird and block
    dim3 blockSize(16);
    dim3 gridSize(ceil(N/((float) blockSize.x)));


    
    /* Matrix Normalization */
    
    matrixNorm<<<gridSize,blockSize>>>(d_A, d_B, N);
    
    // transfer result from device
    cudaMemcpy(B, d_B, sizeof(float)*N*N, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop,0);
    
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    float gpu_elapsed_time_ms = 0.0;
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on GPU: %f ms.\n\n", gpu_elapsed_time_ms);
    printf("\nStopped clock.");
    print_output();
    printf("\n-------------------- CUDA End-------------------------\n");

        // // free both host and device memory
        free(A);
        free(B);
       cudaFree(d_A);
       cudaFree(d_B);
       
       

    // cpu serial computing
    struct timeval start2, stop2;  /* Elapsed times using gettimeofday() */
    struct timezone tzdummy;
    unsigned long long runtime2;
       /* Start Clock */
    printf("\n--------------------Serial Start-----------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");
    gettimeofday(&start2, &tzdummy);
    
    
    /* Matrix Normalization */
    matrixNorm();
    
    
    /* Stop Clock */
    gettimeofday(&stop2, &tzdummy);
    runtime2 = (unsigned long long)(stop2.tv_sec - start2.tv_sec) * 1000000 + (stop2.tv_usec - start2.tv_usec);
    
    
    /* Display timing results */
    printf("Runtime on CPU = %g ms.\n", (float)runtime2/(float)1000);
    printf("\nStopped clock.");
    print_output2();
    printf("\n---------------------Serial End---------------------------\n");
    
    
    exit(0);
}