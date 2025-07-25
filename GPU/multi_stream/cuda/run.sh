
nvcc -O2 -o multi_stream_gemm multi_stream_gemm.cu

nsys profile -t cuda,cudnn,cublas -o multi_stream_gemm ./multi_stream_gemm