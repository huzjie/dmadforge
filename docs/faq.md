# FAQ

**Q: 这是真实可训练的视频模型吗？**
A: 核心是零依赖的参考实现，mock 后端确定性复现"蒸馏收敛"全过程。cpu 后端跑真实的小规模扩散引擎；openai/vllm/transformers 是生产后端适配器。

**Q: 能直接生成视频文件吗？**
A: 本框架输出的是潜变量（latent），实际落地需接入 VAE 解码器。参考 `audiovideo/codec.py` 与 `diffusion/vae.py`。

**Q: 为什么判别器用最近邻而不是 MLP？**
A: 最近邻判别器是 GAN 决策边界的确定性代理，保证收敛曲线干净可复现。真实场景换 MLP/CNN 判别器即可。
