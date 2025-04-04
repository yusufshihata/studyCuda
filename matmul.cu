#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matmul(float* a, float* b, float*c, int n, int m, int k) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < n && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < m; ++i) {
            sum += a[m*row + i] * b[i*k + col];
        }
        c[row*k+col] = sum;
    }
}

void init_matrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = (float)i;
    }
}

void print(float* matrix, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << matrix[i * m + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int n = 3, m = 2, k = 3;
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // Allocate host memory
    h_a = (float*)malloc(sizeof(float) * n * m);
    h_b = (float*)malloc(sizeof(float) * m * k);
    h_c = (float*)malloc(sizeof(float) * n * k);

    init_matrix(h_a, n * m);
    init_matrix(h_b, m * k);

    // Allocate device memory
    cudaMalloc((void**)&d_a, sizeof(float) * n * m);
    cudaMalloc((void**)&d_b, sizeof(float) * m * k);
    cudaMalloc((void**)&d_c, sizeof(float) * n * k);

    // Copy data to device
    cudaMemcpy(d_a, h_a, sizeof(float) * n * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * m * k, cudaMemcpyHostToDevice);

    // Kernel launch
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matmul<<<grid, block>>>(d_a, d_b, d_c, n, m, k);

    // Copy result back
    cudaMemcpy(h_c, d_c, sizeof(float) * n * k, cudaMemcpyDeviceToHost);

    // Print result
    print(h_c, n, k);

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

