CC = nvcc
FLAG = -arch=sm_70

all: batch_matmul

batch_matmul: batch_matmul.cu
	$(CC) -o batch_matmul batch_matmul.cu

clean:
	rm batch_matmul