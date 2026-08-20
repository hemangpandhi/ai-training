# Google Gemma 4-E2B On-Device AI Training & LiteRT Export Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https.mit-license.org)
[![Target Platform](https://img.shields.io/badge/Platform-Android_Automotive_OS_|_Pixel_Tablet-green.svg)]()
[![Inference Engine](https://img.shields.io/badge/Engine-LiteRT_LM_|_OpenCL_GPU-blue.svg)]()

This repository provides an end-to-end, production-ready pipeline for fine-tuning **Google Gemma 4-E2B** models, exporting them to **Google LiteRT (AI Edge Torch)**, fixing FlatBuffer container alignments, and running hardware-accelerated OpenCL GPU inference with **Speculative Decoding (MTP)** on Android devices.

---

## 🌟 Overview

- **Target Device**: Android Automotive OS & Pixel Tablet (OpenCL GPU Delegate / `LITERT_CL`)
- **Model Architecture**: Google Gemma 4-E2B (Text + Audio + Vision multimodal support)
- **Quantization Scheme**: Per-Group INT4 Weights (Group Size = 64) + INT8 Embeddings + INT8 KV-Cache
- **Optimization Target**: **< 380 ms TTFT** (Time To First Token) and **22–25 TPS** (Tokens Per Second) via Multi-Token Prediction (MTP).

---

## 📁 Repository Structure

```
.
├── README.md                              # Complete End-to-End Training & Export Guide
├── requirements.txt                        # Required Python Dependencies
├── dataset/
│   └── sample_in_car_dataset.json          # Standardized intent & slot dataset schema
├── scripts/
│   ├── 01_train_gemma4_e2b.py             # Step 1: Unsloth / PEFT LoRA Fine-Tuning
│   ├── 02_merge_lora_weights.py           # Step 2: Merge LoRA Checkpoint into Full HuggingFace Weights
│   ├── 03_export_litert_lm.py             # Step 3: Export HuggingFace Model to LiteRT TFLite
│   ├── 04_fix_subgraphs_and_signatures.py # Step 4: Fix Subgraph Array & SignatureDef FlatBuffer Offsets
│   ├── 05_build_and_pack_container.py     # Step 5: Assemble model.toml & Pack MTP .litertlm Bundle
│   └── 06_android_ui_automation.py        # Step 6: Automated ADB UI Verification & Model Loader
└── docs/
    ├── ARCHITECTURE_AND_SPECIFICATIONS.md  # Detailed LiteRT Container & Section Specifications
    ├── SPECULATIVE_DECODING_MTP_GUIDE.md   # Multi-Token Prediction (MTP) Drafter Setup
    └── TROUBLESHOOTING_AND_LOGCAT.md       # Diagnostic Guide for FlatBuffer & Tensor Errors
```

---

## ⚡ Quick Start Pipeline

### 1. Installation

```bash
git clone https://github.com/hemangpandhi/ai-training.git
cd ai-training
pip install -r requirements.txt
```

### 2. Phase 1: Fine-Tune Gemma 4-E2B with LoRA

Fine-tune Gemma 4-E2B on domain-specific in-car assistant intents using Low-Rank Adaptation (LoRA):

```bash
python scripts/01_train_gemma4_e2b.py \
  --dataset_path=dataset/sample_in_car_dataset.json \
  --output_dir=in_car_gemma4_e2b_lora \
  --batch_size=2 \
  --epochs=3 \
  --lr=2e-4
```

### 3. Phase 2: Merge LoRA Adapter Weights

Merge the trained LoRA adapters back into base Gemma 4-E2B weights:

```bash
python scripts/02_merge_lora_weights.py \
  --base_model=google/gemma-4-E2B-it \
  --adapter_dir=in_car_gemma4_e2b_lora \
  --output_dir=gemma4_hf_merged
```

### 4. Phase 3: Export to LiteRT TFLite Subgraphs

Export the merged HuggingFace model using `litert-torch`:

```bash
python scripts/03_export_litert_lm.py \
  --model_dir=gemma4_hf_merged \
  --output_dir=litert_export
```

### 5. Phase 4: Re-order FlatBuffer Subgraphs & SignatureDefs

Re-order Section 2 (`tf_lite_prefill_decode.tflite`) subgraphs so that Subgraph 0 is `decode` and SignatureDefs align 1-to-1:

```bash
python scripts/04_fix_subgraphs_and_signatures.py \
  --tflite_path=litert_export/Section2_TFLiteModel_tf_lite_prefill_decode.tflite
```

### 6. Phase 5: Build & Pack `.litertlm` Container

Assemble `model.toml`, sentencepiece tokenizer, embedder, and MTP drafter into the final container bundle:

```bash
python scripts/05_build_and_pack_container.py \
  --bundle_dir=litert_export \
  --output_path=in_car_assistant_gemma4_e2b_pixel_int4.litertlm
```

### 7. Phase 6: Push to Android Device & Automated Testing

Deploy to target Android Automotive OS / Pixel Tablet and verify OpenCL GPU delegate initialization:

```bash
# Push container to Android device via ADB
adb push in_car_assistant_gemma4_e2b_pixel_int4.litertlm /data/local/tmp/

# Run automated Android UI model selection & load test
python scripts/06_android_ui_automation.py
```

---

## 📊 Optimization & Performance Specs

| Metric | Base Model (FP16) | Optimized Container (INT4 + MTP) |
| :--- | :--- | :--- |
| **Model Container Size** | 2.58 GB | **1.80 GB** *(INT8 Embeddings)* |
| **Speculative Decoding** | ❌ None | **✅ 3-Token Parallel MTP Drafter** |
| **TTFT Latency** | ~690 ms | **< 380 ms** |
| **Generation TPS** | ~7.5 TPS | **22 – 25 TPS ⚡** |

---

## 📜 License & Citation

Licensed under the [MIT License](LICENSE).
Created for Google Antigravity & Android On-Device AI Engineering.
