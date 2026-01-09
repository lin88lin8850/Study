# DeepSeek Sparse Attention(DSA)

PyTorch 实现来自 DeepSeek 官方: [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)

官方实现只是 demo 作用，并没有起到加速效果，是在 attention 计算完成后应用的 index_score;
如果要起到加速效果，需要在 cuda 融合 kernel 里面只针对 index 部分的 q 和 k 去计算 attention