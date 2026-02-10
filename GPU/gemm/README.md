# Install & Run
## Install Docker
```
docker build -t 镜像名称:标签 .
```

## Build
```
bash build.sh
```

## Run
```
bash profile.sh
```


# Kernel Analysis
## Background
> * 输入矩阵 A[M, K]，B[K, N]，C[M, N]
> * 输出矩阵 D[M, N]

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


## kernel_1

**关键优化**

1. **Shared Memory Tiling**：将 A、B 的 block tile 先加载到 shared memory，供 warp 复用，大幅减少全局内存访问次数
2. **Tensor Core (mma.sync)**：使用 `mma.sync.aligned.m16n8k8` PTX 指令，配合 `ldmatrix` 从 shared memory 高效加载数据到寄存器，充分利用 A100 Tensor Core 算力

**计算强度**

每个 block 负责计算 $BM \times BN = 256 \times 128$ 的输出 tile，沿 K 方向迭代，每次从全局内存加载：
- A 分块: $BM \times BK \times 2 = 256 \times 128 \times 2$ bytes
- B 分块: $BK \times BN \times 2 = 128 \times 128 \times 2$ bytes

每个 block 全局内存读取总量（忽略 C/D，对大 K 影响不大）：
$$\text{bytes} = \frac{K}{BK} \times (BM + BN) \times BK \times 2 = K \cdot (BM + BN) \cdot 2 = 2K(BM + BN)$$

每个 block 计算量：
$$\text{FLOPs} = 2 \times BM \times BN \times K$$

$$\boxed{\text{计算强度} = \frac{2 \cdot BM \cdot BN \cdot K}{2K(BM + BN)} = \frac{BM \cdot BN}{BM + BN} = \frac{256 \times 128}{384} \approx 85.3 \text{ FLOP/byte}}$$

85.3 FLOP/byte **低于** A100 峰值比（~156 FLOP/byte），kernel_1 仍是**内存带宽受限**

**性能瓶颈**

1. **全局内存加载未向量化**：使用基础 `tileMemcpy`（逐 half 元素拷贝），而非 `tileMemcpyUnrolledVectorized`（128-bit `float4` 向量化加载）
   - 向量化可以减少指令数，充分利用 L1/L2 cache line 提升内存带宽利用率

2. **无 Double Buffering**：当前执行流为 `load tile k → sync → compute tile k → sync → load tile k+1 → ...`，计算和访存串行。
   - 双缓冲使用两份 shared memory buffer，实现计算 tile k 时异步预取 tile k+1

3. **Shared Memory Bank Conflict**：未对 shared memory 布局进行 Swizzle 优化，`ldmatrix` 访问可能产生 bank conflict，降低 shared memory 吞吐量



## kernel_2

**关键优化**

kernel_2 在 kernel_1 的基础上，核心改动是把 A/B 分块从 global memory 搬运到 shared memory 的路径替换为：
1. **向量化加载/存储**：`tileMemcpyUnrolledVectorized`，按 `float4`（128-bit）进行读写
2. **循环展开**：`#pragma unroll`，减少循环控制开销并提升指令级并行（ILP）

**计算强度**

kernel_2 没有改变 block tile 形状（`BM=256, BN=128, BK=128`）， 计算强度和 kernel_1 一致，比 kernel_1 快的原因是：
1. **更少的访存指令**：128-bit 向量化把原本多个标量 half 读写合并，降低指令数与发射压力
2. **更好的内存事务利用率**：更接近 cache line/transaction 粒度，提升有效带宽
3. **更少的循环控制指令和地址计算开销**：展开后减少循环控制指令


## kernel_3

**关键优化**

kernel_3 在 kernel_2 的基础上，核心改动是使用 swizzle 来避免从 share memory 读取 A/B 数据时产生 bank conflict


**计算强度**

kernel_3 计算强度仍未发生改变, 但是由于减少了 bank 冲突，加快了 share memory 的数据读取


## kernel_4

**关键优化**

kernel_4 在 kernel_3 的基础上，核心改动是引入 1-stage pipeline（寄存器缓存预取）：
1. **global -> register cache 预取**：计算当前 `block_k` 时，提前把下一块 A/B tile 通过 `tileMemcpyLoad` 读入寄存器数组（`A_gmem_cache_reg` / `B_gmem_cache_reg`）
2. **register cache -> shared memory 回填**：当前轮计算结束后，再用 `tileMemcpySwizzleStore` 将寄存器中的下一块数据写回 shared memory（保持 swizzle 布局）
3. **计算与访存重叠**：把“下一块 gmem 加载”与“当前块 MMA 计算”重叠，降低 global memory latency 暴露

**计算强度**

kernel_4 计算强度仍未发生改变, 但是引入了流水重叠, 因此执行更快


## kernel_5

**关键优化**

kernel_5 在 kernel_4 的基础上，核心改动是进一步针对 shared memory 访存和 tile 形状做专门化优化：
1. **Block tile 重构（256x128x128 -> 256x256x32）**：增大 N 方向 tile，提升每次加载 A/B 分块后的复用度，从而提高算术强度
2. **专用 `ldmatrix_a/ldmatrix_b` 加载路径**：针对固定的 warp tile 形状（`WM=128, WK=32`）手工展开 `ldmatrix.x4`/`ldmatrix.x4.trans`，减少通用索引计算与指令开销


**计算强度**

kernel_5  `BM=256, BN=256`：
$$\boxed{\text{计算强度}=\frac{2\cdot BM\cdot BN\cdot K}{2K(BM+BN)}=\frac{BM\cdot BN}{BM+BN}=\frac{256\times256}{512}=128\ \text{FLOP/byte}}$$

128 FLOP/byte 仍低于 A100 峰值比（~156 FLOP/byte）；但相较 kernel_4（约 85.3 FLOP/byte）更接近计算上限


## kernel_6

**关键优化**

kernel_6 在 kernel_5 的基础上，核心改动是把原先 1-stage 预取进一步扩展成 **shared memory 双缓冲（ping-pong）**：
1. **双份 shared memory tile**：为 A/B 分块分配两套 buffer，当前 tile 计算时，下一 tile 通过 `tileMemcpyLoad` 预取到寄存器，再回填到另一套 shared memory buffer
2. **buffer 交替切换**：每轮 `block_k` 结束后切换读写 buffer（`offset_direction` 翻转），让“当前轮 MMA 计算”和“下一轮数据准备”重叠更充分

**计算强度**

kernel_6 没有改变 block tile 形状（`BM=256, BN=256, BK=32`），因此理论计算强度与 kernel_5 一致

因此，kernel_6 的加速主要来自**更强的访存-计算重叠**（降低 global memory latency 暴露），而不是算术强度提升
