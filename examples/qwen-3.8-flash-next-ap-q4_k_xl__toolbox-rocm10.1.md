# llama-server benchmark: toolbox-rocm10.1 / Qwen-3.8-Flash-Next-AP-Q4_K_XL

- Date (UTC): 2026-10-25 13:48:20
- URL: http://steelph0enix.framework:44444
- Server build: b1030-26636b02
- Seed: 168061867
- Depths: 0, 8,192, 16,384, 32,768, 65,536, 131,072, 250,000
- Prefill tokens: 10240 / generation tokens: 1024 / repetitions: 3
- Runtime note: generation depth 0 uses a one-token seed prompt

## Model details

Details come from `/props` captured before the sweep.

| Field | Value |
| --- | --- |
| Model | Qwen-3.8-Flash-Next-AP-Q4_K_XL |
| Quantization | IQ4_NL - 4.5 bpw |
| Model path | /home/LLMs/models/qwen/3.8-Flash-Next/AP-Q4_K_XL/Qwen3.8-Flash-Next-AP-Q4_K_XL.gguf |
| Per-slot context | 262,144 |
| llama.cpp build | b1030-26636b02 |

## Prefill

| Depth | Prompt evaluated | Cache reused | Mean prefill tok/s | SD | Mean prefill ms |
| --- | --- | --- | --- | --- | --- |
| 0 | 10240 | 0 | 1370.26 | 17.65 | 7473.86 |
| 8,192 | 10240 | 8192 | 1343.99 | 9.08 | 7619.32 |
| 16,384 | 10240 | 16384 | 1326.86 | 12.75 | 7717.96 |
| 32,768 | 10240 | 32768 | 1302.48 | 12.64 | 7862.44 |
| 65,536 | 10240 | 65536 | 1287.71 | 10.51 | 7952.49 |
| 131,072 | 10240 | 131072 | 1247.64 | 8.83 | 8207.80 |
| 250,000 | 10240 | 250000 | 1166.19 | 5.66 | 8780.88 |

## Generation

| Depth | Predicted tokens | Mean generation tok/s | SD | Mean generation ms |
| --- | --- | --- | --- | --- |
| 0 | 1024 | 31.06 | 0.01 | 32931.53 |
| 8,192 | 1024 | 29.16 | 0.39 | 35081.01 |
| 16,384 | 1024 | 29.10 | 0.11 | 35150.66 |
| 32,768 | 1024 | 28.97 | 0.02 | 35313.82 |
| 65,536 | 1024 | 28.59 | 0.01 | 35775.73 |
| 131,072 | 1024 | 27.88 | 0.00 | 36694.26 |
| 250,000 | 1024 | 26.69 | 0.01 | 38331.69 |
