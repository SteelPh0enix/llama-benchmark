# llama-server benchmark: nix-vulkan / Qwen-3.8-Flash-Next-AP-Q4_K_XL

- Date (UTC): 2026-10-24 23:22:45
- URL: http://steelph0enix.framework:51536
- Server build: b1030-26636b02
- Seed: 1643333656
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
| 0 | 10240 | 0 | 1228.73 | 9.96 | 8334.16 |
| 8,192 | 10240 | 8192 | 1183.67 | 16.26 | 8652.12 |
| 16,384 | 10240 | 16384 | 1156.46 | 15.56 | 8855.70 |
| 32,768 | 10240 | 32768 | 1094.51 | 12.64 | 9356.64 |
| 65,536 | 10240 | 65536 | 989.91 | 10.24 | 10345.12 |
| 131,072 | 10240 | 131072 | 825.11 | 5.48 | 12410.80 |
| 250,000 | 10240 | 250000 | 657.39 | 0.95 | 15576.80 |

## Generation

| Depth | Predicted tokens | Mean generation tok/s | SD | Mean generation ms |
| --- | --- | --- | --- | --- |
| 0 | 1024 | 34.10 | 0.05 | 30002.36 |
| 8,192 | 1024 | 32.45 | 0.15 | 31527.57 |
| 16,384 | 1024 | 32.30 | 0.08 | 31671.25 |
| 32,768 | 1024 | 32.02 | 0.23 | 31951.71 |
| 65,536 | 1024 | 31.69 | 0.08 | 32281.56 |
| 131,072 | 1024 | 30.38 | 0.10 | 33674.22 |
| 250,000 | 1024 | 29.61 | 0.03 | 34553.28 |
