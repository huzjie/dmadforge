# API 参考

## CLI

- `python -m dmadforge doctor` 诊断
- `python -m dmadforge train --epochs N` 蒸馏
- `python -m dmadforge bench` 基准
- `python -m dmadforge serve --port N` HTTP 服务

## HTTP

- `GET /health`
- `GET /v1/models`
- `POST /v1/video/generations`  `{prompt, n_frames, seed}`
- `POST /v1/distill`  `{epochs}`
