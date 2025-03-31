#include <iostream>
#include <cuda_runtime.h>

__global__ void vec_add(float *a, float *b, float *c, const int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    c[idx] = a[idx] + b[idx];
}

void init_vec(float *vec, int n) {
  for (int i = 0; i < n; ++i) {
    vec[i] = i + 1;
  }
}

int main() {
  const int n = 16;
  float *h_a, *h_b, *h_c;
  float *d_a, *d_b, *d_c;

  // Corrected memory allocation
  h_a = (float*)malloc(n * sizeof(float));
  h_b = (float*)malloc(n * sizeof(float));
  h_c = (float*)malloc(n * sizeof(float));

  init_vec(h_a, n);
  init_vec(h_b, n);

  cudaMalloc((void**)&d_a, n * sizeof(float));
  cudaMalloc((void**)&d_b, n * sizeof(float));
  cudaMalloc((void**)&d_c, n * sizeof(float));

  // Move host data to device
  cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
  
  // Kernel launch
  int numThreads = 256;
  int numBlocks = (n + numThreads - 1) / numThreads;
  vec_add<<<numBlocks, numThreads>>>(d_a, d_b, d_c, n);
  cudaDeviceSynchronize();

  // Move the result from device to host (fixed)
  cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; ++i) {
    std::cout << h_c[i] << std::endl; 
  }

  // Free memory
  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}

