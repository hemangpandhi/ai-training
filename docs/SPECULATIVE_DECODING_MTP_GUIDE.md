# Speculative Decoding (MTP) Setup Guide

This guide explains how Multi-Token Prediction (MTP) draft heads work in LiteRT-LM to achieve **22–25 TPS** generation throughput on Android GPUs.

---

## ⚡ How Speculative Decoding Works

1. **Draft Prediction**: In each step, the embedded draft head (`Section11_TFLiteModel_tf_lite_mtp_drafter.tflite`) predicts candidate tokens $t_1, t_2, t_3$ in parallel.
2. **Target Verification**: The main transformer model evaluates all candidate tokens in a single parallel OpenCL GPU forward pass.
3. **Rejection Sampling**: Any candidate token that matches the target distribution is accepted instantly. If a token fails verification, generation reverts to standard target sampling.

---

## 🎯 Verification Guarantee

- **Mathematically Lossless**: Speculative decoding introduces **0.00% quality loss**. The generated text is guaranteed to match standard autoregressive target generation.
- **Latency Reduction**: Reduces OpenCL GPU kernel launch overhead by generating 3 tokens per step.
