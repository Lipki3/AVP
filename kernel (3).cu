#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include <iomanip>  // Для использования манипуляторов вывода
using namespace std;

#define BLOCK_SIZE 16

__global__ void matrixTransform(int* input, int* output, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= M / 2) {
        return;
    }

    // Map 2x2 input window to 4x1 output
    int inputIndex = (2 * row) * M + (col * 2);
    int outputIndex = 4 * row * (M / 2) + col;

    output[outputIndex] = input[inputIndex];
    output[outputIndex + M / 2] = input[inputIndex + 1];
    output[outputIndex + M] = input[inputIndex + M];
    output[outputIndex + 3 * M / 2] = input[inputIndex + M + 1];
}

int main() {
    const int width = 2;
    int N = 60;
    int M = 74;
    int inputSize = N * M;
    int outputSize = N * 2 * (M / 2);

    int* input, * output;
    cudaMallocManaged(&input, inputSize * sizeof(int));
    cudaMallocManaged(&output, outputSize * sizeof(int));

    // Fill input with random values
    for (int i = 0; i < inputSize; i++) {
        input[i] = rand() % 100;
    }

    // Print input
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            cout << setw(width) << input[i * M + j] << " ";
        }
        printf("\n");
    }
    printf("\n\n\n\n");
    // Copy input to device
    cudaMemcpy(input, input, inputSize * sizeof(int), cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((M / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (N * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    matrixTransform << <gridDim, blockDim >> > (input, output, N, M);
    cudaDeviceSynchronize();

    // Print output
    printf("\n\n\n\n\n\n\n\n\n\n\n");
    for (int i = 0; i < 2 * N; i++) {
        for (int j = 0; j < M / 2; j++) {
            cout << setw(width) << output[i * M / 2 + j] << " ";
        }
        printf("\n");
    }

    // Free memory
    cudaFree(input);
    cudaFree(output);

    return 0;
}
