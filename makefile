CC:=gcc
NVCC:=nvcc

program: matrixNorm.c matrixNorm.cu test.cu

	 $(CC) matrixNorm.c -o m1 -g -lm

	 $(NVCC) matrixNorm.cu -o m2 -g -lm
	 $(NVCC) test.cu -o t -g -lm

clean:
	rm -rf *.out
