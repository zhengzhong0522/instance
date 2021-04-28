#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>


__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    
    cuda_hello<<<1,1>>>(); 
    printf("Hello World from CPU!\n");
    cudaError_t err = cudaGetLastError();
    if ( err != cudaSuccess )
    {
       printf("CUDA Error: %s\n", cudaGetErrorString(err));       

       // Possibly: exit(-1) if program cannot continue....
    }
    return 0;
}