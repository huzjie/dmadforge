# 基准口径

四组基准：

1. **speed**：去噪步数 50→4（12.5×），以及墙钟速度比。
2. **quality**：重建 MSE 与音画一致性随 skill 的变化。
3. **distribution**：MMD² 与 Wasserstein-1 到老师分布的距离。
4. **audio**：音频潜变量重建误差与音画同步一致性。

所有基准都在确定性 mock 上运行，结果可复现。
