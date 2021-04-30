#!bin/bash
  
git pull
nvcc  matrixNorm.cu -o m -g -lm
./m 256
./m 512
./m 1024
./m 2048
./m 4096
./m 8192
