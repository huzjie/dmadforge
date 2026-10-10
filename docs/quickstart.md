# 快速开始

```bash
python -m dmadforge doctor
python -m dmadforge train --epochs 20
python -m dmadforge bench
python -m dmadforge serve --port 8765
```

## HTTP 调用

```bash
curl -X POST http://127.0.0.1:8765/v1/video/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a cat walking", "n_frames": 4, "seed": "demo"}'
```
