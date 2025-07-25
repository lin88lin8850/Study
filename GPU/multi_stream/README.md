需要下载 Nsight Systems 软件来看 Trace

# Cuda Kernel 实现多流
可以发现 cuda kernel 实现版本是完全 overlap 的，主要是因为 cpu 流上，先进行了 cudaMalloc() --> kernel lanuch，然后 gpu 上多个流同步开始 kernel 计算

# Torch 实现多流
可以发现 torch 实现版本并没有 overlap，且存在较大空隙，空隙是被 cpu cudaMalloc() 占用，进而导致 cpu kernel lauch delay，再导致 GPU stream kernel 执行的也比较晚

# 建议
后续想要通过多流来进行 overlap，需要控制好各个 kernel lanuch 顺序来保证 overlap  