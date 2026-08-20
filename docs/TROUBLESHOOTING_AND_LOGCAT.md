# Diagnostic & Troubleshooting Guide

This guide details common runtime errors, logcat traces, and resolution steps for LiteRT model deployment on Android devices.

---

## 🐞 Error Matrix & Resolutions

### 1. `The model is not a valid Flatbuffer buffer`
- **Cause**: Multi-bucket embedder subgraphs were missing or corrupted.
- **Fix**: Re-generate flatbuffer or use single-subgraph embedder (`main`) matching Google Gemma 4 specs.

### 2. `Input tensor not found` (`llm_litert_compiled_model_executor_factory.cc:197`)
- **Cause**: Subgraph 0 was pointing to `prefill_128` instead of `decode`.
- **Fix**: Execute `scripts/04_fix_subgraphs_and_signatures.py` to re-order the FlatBuffer subgraph array so Subgraph 0 is `decode`.

### 3. `Model requires one of [gpu_artisan] but Main backend is GPU`
- **Cause**: `backend_constraint = "gpu_artisan"` was set in `model.toml` for a `prefill_decode` model section.
- **Fix**: Remove restrictive `backend_constraint` or use standard `model_type = "prefill_decode"`.

---

## 🔍 Logcat Monitoring Commands

```bash
# Capture full LLMManager and LiteRT logcat stream
adb logcat -v time LLMManager:V LiteRT:V litert_lm:V tflite:V *:S

# Check successful engine initialization line:
# LLM Initialized successfully from /data/local/tmp/in_car_assistant_gemma4_e2b_pixel_int4.litertlm
```
