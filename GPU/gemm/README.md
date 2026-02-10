# Install & Run
## 安装镜像
```
docker build -t 镜像名称:标签 .
```

## 编译和安装
```
bash build.sh
```

## Run
```
bash profile.sh
```

# Kernel 分析
## Background
> 1). 输入矩阵 A[M, K]，B[K, N]，C[M, N]  
> 2). 输出矩阵 D[M, N]

A100 内存带宽 2TB/s，FP16 算力是 312 TFLOPS

$$\text{A100 理论峰值性能} = \frac{312 \text{ TFLOPS}} {2 \text{TB/s}} \approx 156 \text{ FLOP/byte}$$

## kernel_0
**性能分析**  
kernel_0 是一个 naive gemm 实现，每个线程负责计算一个输出元素:
- 读 A (全局内存 → L2/L1 cache → 寄存器): K 次
- 读 B (全局内存 → L2/L1 cache → 寄存器): K 次
- 读 C (全局内存 → L2/L1 cache → 寄存器): 1 次
- 写 D (寄存器 → L1/L2 cache → 全局内存): 1 次  

$$ 计算强度 = \frac{2K}{2*(2K+2)} = \frac{K}{2K+2} \approx 0.5 \text{ FLOP/byte}$$

kernel_0 计算强度仅 **0.5 FLOP/byte**, **远低于峰值**, 是内存受限，而非计算受限