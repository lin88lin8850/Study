# Engram Architecture Demo Implementation

PyTorch 实现来自 DeepSeek 官方: [Engram](https://github.com/deepseek-ai/Engram/tree/main)

1. Demo Purpose Only: 
   This code is a demonstration version intended to illustrate the core logic and 
   data flow of the Engram module.

2. Production Readiness: 
   This implementation requires further optimization for actual production use 
   (e.g., custom CUDA kernels, distributed training support).

3. Simplifications: 
   Standard components (Normalization, Attention, MoE) and complex Hyper-connection 
   mechanisms are omitted or mocked in this version to focus exclusively on the 
   Engram module implementation.

# Run
```
pip3 install -r requirements.txt

python3 engram_demo_v1.py
```