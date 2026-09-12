# EMLM: Electromagnetic Large Language Model

This repository provides the implementation of
EMLM. It converts raw complex I/Q samples into continuous signal tokens and
conditions a language model to answer electromagnetic-signal questions.

Two inference variants are implemented:

| Variant | Intended use | IQ length | Signal-language interface |
|---|---|---:|---|
| EMLM_Recognition | Four-task open-ended recognition | 128 to 5,000,000 (validated lengths below) | 16 question-conditioned signal queries |
| EMLM_Description | Long-form electromagnetic signal description | 16 to 8,000,000 | continuous 1M-point chunk encoding and 64 question-conditioned signal queries |

## Repository layout

```text
.
├── configs/                 
├── emlm/                    
├── infer.py                
└── requirements.txt
```

## Requirements

The reference runtime was validated with Python 3.13, CUDA on NVIDIA H100,
PyTorch 2.9.1, Transformers 5.13.1, and Flash Linear Attention 0.5.1. Other
CUDA accelerators and dependency versions have not yet been validated.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If TileLang needs an explicit CUDA toolchain, set it before inference:

```bash
export EMLM_TOOLCHAIN=/path/to/cuda-toolchain
```

## Weights

EMLM_Recognition: https://drive.google.com/file/d/1MV4rREcGGVfflDzI6g5qpnfjMsoQIMd1/view?usp=drive_link
EMLM_Description: https://drive.google.com/file/d/13P4pv_loh59yWEeF8Y29EZHiPk2OyZG_/view?usp=drive_link

After downloading the weights, create a local `weights/` directory and
place them at:

```text
weights/EMLM_Recognition_inference.pt
weights/EMLM_Description_inference.pt
```

## Input format

Both variants accept:

- a NumPy `.npy` signal with real shape `[L, 2]` or complex shape `[L]`, or a
  two-column `.csv` file;
- a sampling rate in Hz; EMLM_Description requires a finite positive value;
- acquisition-setting text for EMLM_Description;
- a natural-language question.

EMLM_Recognition additionally requires a task adapter name:

- `amc`: modulation recognition;
- `interference`: interference recognition;
- `uav`: UAV-device recognition;
- `wtc`: wireless-technology recognition.

EMLM_Recognition has been validated at lengths `128`, `256`, `976`, `1024`, `4096`,
`16368`, and `5000000`. EMLM_Description accepts any length from `16` through
`8000000`.

## Quick start

EMLM_Recognition:

```bash
python infer.py \
  --model EMLM_Recognition \
  --config configs/EMLM_Recognition.example.json \
  --signal /path/to/iq.npy \
  --sample-rate 20000000 \
  --task wtc \
  --question "这段信号采用了哪种无线通信技术？"
```

EMLM_Description:

```bash
python infer.py \
  --model EMLM_Description \
  --config configs/EMLM_Description.example.json \
  --signal /path/to/iq.npy \
  --sample-rate 20000000 \
  --input-setting "信号采样率为20 MHz，观测时长由IQ点数计算。" \
  --question "请描述这段信号的关键时域、频域和调制特征。"
```


