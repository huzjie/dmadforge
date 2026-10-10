# 架构与数据流

## 组件关系

```
Prompt ──> StudentGenerator (4-step, LoRA) ──> video/audio latents
                    ▲
                    │ 分布匹配 (adversarial + MMD + recon)
                    │
TeacherDiffusion (50-step) ──> gold latents (目标分布)
                    │
                    ▼
              Discriminator (判别 老师/学生)
```

## 蒸馏数据流

1. 老师用 50 步 DDIM 采样出 `gold` 潜变量（目标分布 P）。
2. 学生用 4 步采样出 `student` 潜变量（当前分布 Q）。
3. 判别器吃两者，输出"是不是老师"的概率。
4. 损失 = `adv_weight * 对抗损失 + mmd_weight * MMD² + recon_weight * MSE`。
5. 反向传播只更新学生的 LoRA 因子（A、B），底模冻结。

## 关键设计决策

- **单共享潜空间**：视频帧和音频帧投影到同一个 `latent_dim` 宽度，共用一条去噪主干，音画同步由共享注意力隐式学到。
- **判别器只用 MLP**：判别目标不是整张图，而是"这个潜变量分布对不对"，轻量 MLP 足够。
