# LoRA 与参数账

## 低秩适配

`W' = W + (alpha / rank) * A @ B`，其中 `A ∈ R^{in×rank}`，`B ∈ R^{rank×out}`。

## 参数账（33B 场景）

- 底模：~33B 参数，冻结。
- LoRA rank=128：`in*rank + out*rank`。若 `in=out=4096`，则 `4096*128*2 ≈ 1.05M` 参数/层。
- 全模型 LoRA 总量 ~1.4GB（fp32），远小于底模。

## 复用要点

- 秩越高容量越大，但内存/算力也涨；rank=128 是论文的甜点。
- `alpha` 通常设为 `rank`，缩放系数 = 1。
