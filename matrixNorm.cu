#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

/* Program Parameters */
int N = 6000;

/* Matrices */
float *A, *B;


/* Initialize A and B*/
void initialize_inputs() {
    int row, col;
    
    srand((unsigned)time(NULL));
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[row*N+col] = (float)rand() / 32768.0;
            B[row*N+col] = 0.0;
        }
    }
    
}


/* device function */


__global__ void matrixNorm(float *d_a, float *d_b, int n) {
    // get thread id(col) in the grid
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int mu, sigma, row;
    mu = 0.0;
    for(row=0;row<n;row++){
        mu += d_a[row*n + col];
    }
    mu /= n;

    // make sure the calculation of mean is completed
    __syncthreads();

    sigma = 0.0;
    for (row=0; row < n; row++){
        sigma += powf(d_a[row*n+col] - mu, 2.0);
    }   
        sigma /= n;
    // make sure the calculation of standard deviation is completed
    __syncthreads();

    sigma = sqrt(sigma);

    for(row=0;row<n;row++){
        if(sigma==0.0)
            d_b[row*n+col]=0.0;
        else
        d_b[row*n+col] = (d_a[row*n+col] - mu) / sigma;
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

    
    // copy the host data to device
    cudaMemcpy((void*)d_A, (void*)A, 4*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_B, (void*)B, 4*N*N, cudaMemcpyHostToDevice);
    
    // set up dimension of grid and block, 1-dim gird and block
    dim3 blockSize(32);
    dim3 gridSize((N+blockSize.x - 1) / blockSize.x);

    /* Start Clock */
    printf("\n---------------------------------------------\n");
    printf("Matrix size N = %d", N);
    printf("\nStarting clock.\n\n");
    float gpu_elapsed_time_ms;

    // some events to count the execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start to count execution time of GPU version
    cudaEventRecord(start, 0);
    
    
    /* Matrix Normalization */
    
    matrixNorm<<<gridSize,blockSize>>>(d_A, d_B, N);
    
    // transfer result from device
    cudaMemcpy(B, d_B, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

    // free both host and device memory
    free(A);
    free(B);
    cudaFree(d_A);
    cudaFree(d_B);
    
    /* Stop Clock */
    cudaThreadSynchronize();
    // time counting terminate
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // compute time elapse on GPU computing
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on GPU: %f ms.\n\n", gpu_elapsed_time_ms);
    printf("\nStopped clock.");
    printf("\n---------------------------------------------\n");
    
    exit(0);
}