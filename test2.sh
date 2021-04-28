#!bin/bash

git pull
gcc matrixNorm.c -o m1 -g -lm
nvcc  matrixNorm.cu -o m2 -g -lm
nvcc test.cu -o t -g -lm 
