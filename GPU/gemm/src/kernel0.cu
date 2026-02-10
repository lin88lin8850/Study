#include <cuda.h>
#include <mma.h>

#include "device_utils.cuh"
#include "structs_n_stuff.cuh"

/**
 * kernel_0: Naive GEMM kernel (no shared memory, no TensorCore)
 *
 * 每个线程计算一个输出元素 D[m,n]
 * 从全局内存直接读取，逐个累加
 *
 * 计算: D = alpha * (A @ B) + beta * C
 * 其中 A 是 M×K, B 是 K×N, C/D 是 M×N
 */
__global__ void kernel_0(half* A, half* B, half* C, half* D, const float alpha,
                         const float beta, const unsigned int M,
                         const unsigned int N, unsigned int K) {
  // 每个线程对应输出矩阵 D 中的一个元素
  const unsigned int m = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;

  // 边界检查
  if (m >= M || n >= N) return;

  // 初始化累加器
  float acc = 0.0f;

  // K 循环：逐个累加 A[m, k] * B[k, n]
  for (unsigned int k = 0; k < K; k++) {
    half a_val = A[m * K + k];
    half b_val = B[k * N + n];

    // half 转 float 做乘法，再累加
    acc += __half2float(a_val) * __half2float(b_val);
  }

  // 写回结果
  D[m * N + n] =
      __float2half(alpha * acc + (beta * __half2float(C[m * N + n])));
}

/**
 * kernel_0_launch: 启动 kernel_0
 */
void kernel_0_launch(sgemm_params device_sgemm_params, KernelLogger& timer,
                     const unsigned int num_runs = 10) {
  const unsigned int M = device_sgemm_params.M;
  const unsigned int N = device_sgemm_params.N;
  const unsigned int K = device_sgemm_params.K;

  // 简单的线程块配置：16×16 输出 tile
  constexpr unsigned int BLOCK_M = 16;
  constexpr unsigned int BLOCK_N = 16;

  dim3 blockDim(BLOCK_N, BLOCK_M);
  dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

  for (int i = 0; i < num_runs; i++) {
    timer.Start();
    kernel_0<<<gridDim, blockDim>>>(
        device_sgemm_params.A, device_sgemm_params.B, device_sgemm_params.C,
        device_sgemm_params.D, device_sgemm_params.alpha,
        device_sgemm_params.beta, M, N, K);
    timer.Stop();
  }

  double gflops_per_sec = timer.logKernelStats(M, N, K);
  std::cout << gflops_per_sec << " GFLOPS/sec for " << M << "x" << N << "x" << K
            << std::endl;
  CUDA_CHECK(cudaPeekAtLastError());
}