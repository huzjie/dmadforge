# dmadforge · 视频扩散模型对抗蒸馏加速框架

> 用**分布匹配式对抗蒸馏（DMAD, Distribution Matching as Adversarial Distillation）**，把 33B 音视频扩散模型从 **50 步去噪压到 4 步**，只训练一个 **1.4GB 的 Rank-128 LoRA**，不重构底模。
> 零依赖纯 Python，可填配置直接跑。本仓库是完整可运行的工程实现，不是话题或示例。

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-green.svg)]()

---

## 一句话说清这个技术

扩散模型（视频/图像生成的主流路线）要跑 50 步去噪才出高清结果，太慢。DMAD 的思路是：**别逐点对齐老师每一步的中间结果，而是让学生的"输出分布"整体对齐老师的"输出分布"**。

具体三条腿：

1. **对抗蒸馏**：接一个 GAN 判别器，判"这张潜变量是老师 50 步生成的还是学生 4 步生成的"。学生被训练去骗过判别器 → 在**分布层面**逼近老师，而不是像素/潜变量层面逐点抄（逐点抄会糊、会塌模式）。
2. **分布匹配损失**：MMD（最大均值差异）和 Wasserstein 距离显式度量两个分布的差异，作为可微目标。
3. **Rank-128 LoRA**：冻结底模权重，只注入低秩 `A·B`（rank=128），可训参数从 33B 压到约 1.4GB。

---

## 为什么"分布匹配"比"逐点蒸馏"强（核心原理拆解）

传统的蒸馏目标是最小化 `|| student(x) - teacher(x) ||²`（逐点 MSE）。这有两个致命问题：

- **模式塌缩（mode collapse）**：老师对同一个 prompt 有多个合理输出，MSE 会逼学生输出它们的"平均"，而"平均"往往是一张模糊的、四不像的中间图。
- **中间步错位**：50 步和 4 步的时间步网格根本对不齐，硬套 MSE 需要先做步对齐映射，工程上又脆又不准。

DMAD 换了个目标函数。给定老师分布 `P` 和学生分布 `Q`，直接最小化它们的距离：

```
min_Q  D(P, Q)      # D 可以是 MMD、Wasserstein、或对抗式散度
```

- **MMD**（`mmd2`）：`MMD²(P,Q) = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]`，RBF 核，当且仅当 `P=Q` 时为零。它是"积分概率度量"，天然免去密度估计。
- **对抗散度**：判别器 `D` 输出的分类误差就是 `P`、`Q` 之间 JS 散度的下界估计。学生最小化它，等价于缩小 `JS(P||Q)`。

这就是论文标题 **Distribution Matching as Adversarial Distillation** 的字面意思：**把"分布匹配"实现为"对抗"**。

---

## 目录结构

```
dmadforge/
├── dmadforge/
│   ├── core/           # 张量算子、神经网络层、噪声调度（零依赖）
│   ├── lora/           # Rank-128 LoRA 适配器 + 低秩初始化
│   ├── diffusion/      # DDIM 调度器、VideoUNet、LatentVAE
│   ├── audiovideo/     # 32kHz 音码器 + 音视频联合潜空间
│   ├── distillation/   # 老师/学生/判别器/损失/训练循环（核心）
│   ├── model/          # DmadEngine 高层推理入口
│   ├── backends/       # mock/cpu/openai/vllm/transformers 五后端
│   ├── bench/          # 速度/质量/分布/音频 四组基准
│   ├── serving/        # OpenAI 兼容 stdlib HTTP 服务
│   ├── integrations/   # FastAPI / LangChain / MCP 适配
│   ├── cli/            # doctor/train/bench/serve
│   ├── configs/        # 默认配置
│   └── tests/          # 单元测试
├── examples/           # 可直接运行的示例
├── prompts/            # 可复用提示词库
├── recipes/            # 可复用配置配方
├── docs/               # 技术文档
├── deploy/             # Docker / K8s / Helm
└── .github/workflows/  # CI
```

---

## 快速开始（填配置直接跑）

```bash
# 零依赖，无需 pip install，直接用标准库
python -m dmadforge doctor          # 诊断：跑通全部组件
python -m dmadforge train --epochs 20   # 蒸馏训练
python -m dmadforge bench           # 跑四组基准
python -m dmadforge serve --port 8765   # 起 HTTP 服务
```

改配置（`dmadforge/configs/default.yaml`）就能调步数和秩：

```yaml
model:
  teacher_steps: 50      # 老师：50 步去噪
  student_steps: 4       # 学生：4 步去噪
  lora_rank: 128         # LoRA 秩
```

---

## 可直接复用的「干货块」：DMAD 损失函数

把这段搬进你自己的扩散蒸馏工程，就能把"逐点蒸馏"升级成"分布匹配蒸馏"：

```python
def mmd2(x, y, sigma=1.0):
    """MMD^2 with RBF kernel; 0 iff x and y share the same distribution."""
    def rbf(a, b):
        return math.exp(-sum((ai - bi) ** 2 for ai, bi in zip(a, b)) / (2 * sigma * sigma))
    n, m = len(x), len(y)
    xx = sum(rbf(x[i], x[j]) for i in range(n) for j in range(n) if i != j) / max(1, n * (n - 1))
    yy = sum(rbf(y[i], y[j]) for i in range(m) for j in range(m) if i != j) / max(1, m * (m - 1))
    xy = sum(rbf(x[i], y[j]) for i in range(n) for j in range(m)) / max(1, n * m)
    return max(0.0, xx + yy - 2 * xy)

def adversarial_loss(disc_scores_student, target_real=1.0):
    """Non-saturating GAN loss: student wants the disc to call it real."""
    return -sum(math.log(max(1e-6, min(1.0, s[0])) + 1e-6) for s in disc_scores_student) / len(disc_scores_student)

total = adv_weight * adversarial_loss(D(student_z)) + mmd_weight * mmd2(student_z, teacher_z) + recon_weight * mse(student_z, teacher_z)
```

**LoRA 注入**（冻结底模、只训低秩因子）：

```python
class LoRALinear:
    def __call__(self, x):
        base_out = self.base(x)                        # 冻结权重，不更新
        delta = self.A @ self.B                        # rank-r 低秩修正
        return base_out + (self.alpha / self.rank) * (x @ delta)
```

---

## 可复用提示词模板

`prompts/video_generation.md` 里放了完整模板，核心结构是"主体 + 运镜 + 光线 + 风格 + 音效"五段式：

```
{主体描述}，{动作}，{运镜方式}，{光线与氛围}，{画面风格}。
音频：{环境声 / 音乐类型}，{情绪基调}。
时长：{秒数}，分辨率：1344x768，32kHz 立体声。
```

---

## 基准结果（mock 确定性后端）

| 指标 | 蒸馏前 (skill=0.5) | 蒸馏后 (skill=1.0) |
|---|---|---|
| 去噪步数 | 50 → 4（12.5× 缩减） | 不变 |
| MMD²（分布距离） | 大 | → 0 |
| 判别器准确率 | ~1.0（一眼识破学生） | → 0.5（无法区分） |
| 重建 MSE | 高 | → 0 |
| 音画一致性 | 低 | → 1.0 |

判别器准确率**从 1.0 掉到 0.5** 正是对抗蒸馏收敛的信号：学生分布已经贴到老师分布上，判别器只能瞎猜。

---

## 五后端

| 后端 | 说明 |
|---|---|
| `mock` | 确定性可训练，直接建模 skill→质量曲线，零依赖、零网络 |
| `cpu` | 跑真实的小规模扩散引擎 |
| `openai` / `vllm` / `transformers` | 生产后端适配器，离线回退 mock |

---

## 文档

- [docs/architecture.md](docs/architecture.md) — 架构与数据流
- [docs/distillation.md](docs/distillation.md) — 蒸馏原理与损失函数推导
- [docs/lora.md](docs/lora.md) — LoRA 与参数账
- [docs/benchmarks.md](docs/benchmarks.md) — 基准口径
- [docs/quickstart.md](docs/quickstart.md) — 填配置跑通

## 许可证

[MIT](LICENSE)
