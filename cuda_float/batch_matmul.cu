#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16 // Define block size for CUDA

// Kernel function to perform matrix multiplication for each batch
__global__ void batchMatMul(float* A, float* B, float* C, int N, int batch_size) {
    int batch_idx = blockIdx.z; // Batch index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row of the matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column of the matrix
    
    if (row < N && col < N) {
        float value = 0;
        for (int k = 0; k < N; ++k) {
            value += A[batch_idx * N * N + row * N + k] * B[batch_idx * N * N + k * N + col];
        }
        C[batch_idx * N * N + row * N + col] = value;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void loop(int N, int batch_size) {
    
    // Host memory allocations
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(batch_size * N * N * sizeof(float));
    h_B = (float*)malloc(batch_size * N * N * sizeof(float));
    h_C = (float*)malloc(batch_size * N * N * sizeof(float));

    // Initialize host matrices with some values
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < N * N; ++i) {
            h_A[b * N * N + i] = static_cast<float>(rand() % 10);
            h_B[b * N * N + i] = static_cast<float>(rand() % 10);
        }
    }

    // Device memory allocations
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void**)&d_A, batch_size * N * N * sizeof(float)), "Failed to allocate device memory for A");
    checkCudaError(cudaMalloc((void**)&d_B, batch_size * N * N * sizeof(float)), "Failed to allocate device memory for B");
    checkCudaError(cudaMalloc((void**)&d_C, batch_size * N * N * sizeof(float)), "Failed to allocate device memory for C");


    // Create CUDA events to measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // Copy data from host to device
    checkCudaError(cudaMemcpy(d_A, h_A, batch_size * N * N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_B, h_B, batch_size * N * N * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy B to device");

    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate and print execution time
    float t_transfer = 0;
    cudaEventElapsedTime(&t_transfer, start, stop);
   
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE, batch_size);

    // Create CUDA events to measure time
    //cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // Launch kernel
    batchMatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N, batch_size);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");

    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate and print execution time
    float t_execution = 0;
    cudaEventElapsedTime(&t_execution, start, stop);
    

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, batch_size * N * N * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy C to host");

    // Print result
    /*
    for (int b = 0; b < batch_size; ++b) {
        std::cout << "Result for batch " << b << ":\n";
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << h_C[b * N * N + i * N + j] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    */

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << batch_size <<", " << N << ", " << t_transfer << ", " << t_execution << std::endl;
    

    return;
}

int main()
{
    int batch_sizes[6] = {1, 2, 4, 8, 16, 32};
    int shapes[4] = {1000, 2000, 3000, 4000};
    std::cout << "bs, shape, transfer, compute" << std::endl;
    for(auto batch_size: batch_sizes)
    {
        for(auto shape: shapes)
        {
            loop(shape, batch_size);
        }
    }
    return 0;
}