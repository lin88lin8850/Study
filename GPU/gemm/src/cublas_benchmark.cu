#include <cublas_v2.h>

#include "structs_n_stuff.cuh"

void cublas_launch(sgemm_params device_sgemm_params, KernelLogger& timer,
                   const unsigned int num_runs = 10) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  const half alpha = device_sgemm_params.alpha;
  const half beta = device_sgemm_params.beta;
  const int M = device_sgemm_params.M;
  const int N = device_sgemm_params.N;
  const int K = device_sgemm_params.K;

  // warmup + 计算正确的值, 存入 D 矩阵, 用于后续的精度验证
  cublasStatus_t status = cublasHgemm(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, device_sgemm_params.B,
      N, device_sgemm_params.A, K, &beta, device_sgemm_params.D, N);

  for (int i = 0; i < num_runs; i++) {
    timer.Start();
    status = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                         device_sgemm_params.B, N, device_sgemm_params.A, K,
                         &beta, device_sgemm_params.C, N);
    timer.Stop();

    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("cuBLAS kernel failed");
    }
  }
  double gflops_per_sec = timer.logKernelStats(M, N, K);
  std::cout << "cuBLAS: " << gflops_per_sec << " GFLOPS/sec for " << M << "x"
            << N << "x" << K << std::endl;
}
