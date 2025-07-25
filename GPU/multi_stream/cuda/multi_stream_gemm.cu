#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// 简单的 GEMM kernel
__global__ void gemm_kernel(const float* A, const float* B, float* C, int N, int row_offset) {
    int row = blockIdx.x * blockDim.x + threadIdx.x + row_offset;
    if (row < N) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    const int N = 1024; // 矩阵大小 NxN
    const int num_streams = 4;
    const int block_size = N / num_streams;

    // 分配主机和设备内存
    std::vector<float> h_A(N * N), h_B(N * N), h_C(N * N, 0), h_C_ref(N * N, 0);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

    // 创建 streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // 计时
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));

    // 多流分块 launch kernel
    int threads = 128;
    int blocks_per_stream = (block_size + threads - 1) / threads;
    for (int i = 0; i < num_streams; ++i) {
        int row_offset = i * block_size;
        gemm_kernel<<<blocks_per_stream, threads, 0, streams[i]>>>(d_A, d_B, d_C, N, row_offset);
    }

    // 等待所有 stream 完成
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    std::cout << "Multi-stream GEMM overlap time: " << ms << " ms" << std::endl;

    // 拷贝结果回主机
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    // 校验正确性（CPU 参考实现）
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += h_A[row * N + k] * h_B[k * N + col];
            }
            h_C_ref[row * N + col] = sum;
        }
    }
    float max_diff = 0.0f;
    for (int i = 0; i < N * N; ++i) {
        max_diff = std::max(max_diff, std::abs(h_C[i] - h_C_ref[i]));
    }
    std::cout << "Max diff: " << max_diff << std::endl;

    // 释放资源
    for (int i = 0; i < num_streams; ++i) {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    return 0;
}