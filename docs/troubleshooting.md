# 排障

## 判别器准确率不是 0.5

正常。判别器准确率从 1.0 掉到 0.5 是收敛信号，不是 bug。

## 改了 config 不生效

确认路径指向 `dmadforge/configs/default.yaml`，且 YAML 缩进正确。托管 Python 无 PyYAML 时会走内置 yamlish 子集解析器。

## 想调步数/秩

改 `model.student_steps` 和 `model.lora_rank`，参考 `recipes/` 里的配方。
