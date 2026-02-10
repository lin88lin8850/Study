#!/bin/bash

set -e  # 任何命令失败时退出

# 检查 runner 是否存在
if [ ! -f "./build/runner" ]; then
    echo "Error: build/runner not found! Running build.sh..."
    ./build.sh
fi

# GPU 配置（可选，需要 sudo）
# Enable persistence mode to keep GPU initialized
# nvidia-smi -pm 1

# # For A100, lock the [GPU|Memory] clocks to the maximum supported values
# nvidia-smi --lock-gpu-clocks=1410
# nvidia-smi --lock-memory-clocks=1593
# nvidia-smi -ac 1593,1410

export CUDA_VISIBLE_DEVICES=4

# 定义超时时间（秒）
TIMEOUT=300

# 辅助函数：带超时的执行
run_kernel() {
    local kernel_id=$1
    local num_runs=$2
    local m=$3
    local n=$4
    local k=$5
    local check_acc=$6
    local label=$7

    echo "Running: $label (M=$m, N=$n, K=$k, runs=$num_runs)"
    timeout $TIMEOUT ./build/runner $kernel_id $num_runs $m $n $k $check_acc || {
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "⚠ Timeout! Kernel $kernel_id took longer than ${TIMEOUT}s"
        else
            echo "⚠ Kernel $kernel_id failed with exit code $exit_code"
        fi
    }
}


# 1. 验证算子精度，使用小尺寸（CPU GEMM 大尺寸计算很耗时）
M=256
N=128
K=128
NUM_ITERATIONS=1
CHECK_ACC=true
run_kernel 0 $NUM_ITERATIONS $M $N $K $CHECK_ACC "kernel_1"
# run_kernel 99 $NUM_ITERATIONS $M $N $K $CHECK_ACC "cuBLAS"
echo "------------------------------"

# 2. 性能测试
M=1024
N=1024
K=1024
NUM_ITERATIONS=30
CHECK_ACC=false
run_kernel 0 $NUM_ITERATIONS $M $N $K $CHECK_ACC "kernel_0"
run_kernel 1 $NUM_ITERATIONS $M $N $K $CHECK_ACC "kernel_1"
run_kernel 2 $NUM_ITERATIONS $M $N $K $CHECK_ACC "kernel_2"
run_kernel 3 $NUM_ITERATIONS $M $N $K $CHECK_ACC "kernel_3"
run_kernel 4 $NUM_ITERATIONS $M $N $K $CHECK_ACC "kernel_4"
run_kernel 5 $NUM_ITERATIONS $M $N $K $CHECK_ACC "kernel_5"
run_kernel 6 $NUM_ITERATIONS $M $N $K $CHECK_ACC "kernel_6"
run_kernel 99 $NUM_ITERATIONS $M $N $K $CHECK_ACC "cuBLAS"
echo "------------------------------"

echo "✓ Complete!"