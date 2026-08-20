# Complete Troubleshooting, Diagnostics & Logcat Reference

This document provides a comprehensive diagnostic reference covering all runtime errors, root causes, FlatBuffer binary fixes, and resolution procedures encountered when training, exporting, and deploying **Google Gemma 4-E2B** models on Android Automotive OS and Pixel Tablets (`LITERT_CL` GPU Delegate).

---

## 🛠️ Complete Issue & Resolution Matrix

### Issue 1: `The model is not a valid Flatbuffer buffer` (`model_data.cc:437`)
- **Symptom**: Runtime initialization failure when loading `.litertlm` container on Android.
- **Root Cause**: Missing or truncated multi-bucket subgraphs in `Section 2` (`embedder`) or `Section 3` (`per_layer_embedder`).
- **Resolution**: Ensure `embedder.tflite` and `per_layer_embedder.tflite` contain **EXACTLY 1 SUBGRAPH** (`Subgraph 0: main`) with single input/output tensors (`embedder_token_ids:0 -> StatefulPartitionedCall:0`).

---

### Issue 2: `Model requires one of [gpu_artisan] but Main backend is GPU`
- **Symptom**: Strict backend mismatch error when selecting `GPU` in app dropdown.
- **Root Cause**: `backend_constraint = "gpu_artisan"` was declared in `model.toml` under `[[section]] model_type = "prefill_decode"`.
- **Resolution**: Remove restrictive `backend_constraint` from `model.toml`. Standard OpenCL GPU initialization passes backend type `"gpu"`.

---

### Issue 3: `Input tensor not found` (`llm_litert_compiled_model_executor_factory.cc:197`)
- **Symptom**: LiteRT-LM C++ loader fails during executor creation.
- **Root Cause**: Subgraph 0 of `Section10_TFLiteModel_tf_lite_prefill_decode.tflite` was `prefill_128` instead of `decode`.
- **Resolution**: Run `scripts/04_fix_subgraphs_and_signatures.py` to re-order FlatBuffer subgraphs so Subgraph 0 is `decode`, Subgraph 1 is `prefill_4096`, etc.

---

### Issue 4: `Signature has incorrect number of input/outputs` (`model_load.cc:396`)
- **Symptom**: Model load fails with signature mismatch error.
- **Root Cause**: SignatureDef array pointed key `'decode'` to `prefill_128` (30 outputs instead of 31 outputs with `logits`).
- **Resolution**: Re-align SignatureDefs so `key='decode'` maps to Subgraph 0 (`decode`) and sequence bucket keys map to their corresponding prefill subgraphs.

---

### Issue 5: SignatureDef `subgraph_index` vtable offset fallback to 0
- **Symptom**: SignatureDefs key `'prefill_4096'` or `'decode'` silently fallback to Subgraph 0.
- **Root Cause**: FlatBuffers omits default values (0) in vtable offsets, causing `SignatureDef.SubgraphIndex()` to return 0 for all keys.
- **Resolution**: Use `scripts/04_fix_subgraphs_and_signatures.py` to explicitly set table offsets and align SignatureDef indices.

---

### Issue 6: Unprefixed vs Prefixed Tensor Names (`decode_kv_slice_0` vs `kv_cache_0`)
- **Symptom**: `litert-torch` exports prefixed names like `decode_kv_slice_0_output` which the C++ runtime fails to bind.
- **Root Cause**: Default export function lacks tensor canonicalization.
- **Resolution**: Run `scripts/00_patch_litert_torch.py` to apply `_canonicalize_tensor_name()` to `litert_converter.py`.

---

### Issue 7: JAX Tensor Shape Padding Mismatch (`_aten_add_tensor`)
- **Symptom**: Tensor rank match but dimension shape mismatch during MLIR ATen lowerings.
- **Root Cause**: JAX lowering rules for `torch.ops.aten.add.Tensor` lack automatic padding for sequence broadcast dimensions.
- **Resolution**: Run `scripts/00_patch_litert_torch.py` to apply shape padding logic to `lowerings.py`.

---

### Issue 8: Speculative Decoding MTP Drafter Alignment (`Section 11`)
- **Symptom**: MTP speculative decoding fails to initialize or falls back to single-token mode.
- **Root Cause**: Missing `model_type = "mtp_drafter"` declaration in `model.toml` or missing `Section11_TFLiteModel_tf_lite_mtp_drafter.tflite`.
- **Resolution**: Run `scripts/05_build_and_pack_container.py` to embed the 30-subgraph MTP drafter binary and update `model.toml`.

---

### Issue 9: Android UI Automation Center-Click Selection
- **Symptom**: UI tapping misses dropdown items or backend selector.
- **Root Cause**: Hardcoded screen coordinates vary across screen resolutions and Automotive OS display density.
- **Resolution**: Use `scripts/06_android_ui_automation.py` which dumps Android UI XML (`uiautomator dump`), parses exact node bounding boxes `[x1,y1][x2,y2]`, and calculates dynamic center coordinates `((x1+x2)/2, (y1+y2)/2)`.

---

## 🔍 Logcat Monitoring & Verification Stream

To monitor device execution in real-time on Android Automotive OS or Pixel Tablet:

```bash
# Filter logcat for LLMManager, tflite, and LiteRT tags
adb -s 3704105H8094TU shell "logcat -v time LLMManager:V LiteRT:V litert_lm:V tflite:V *:S"
```

### ✅ Expected Logcat Output for Successful GPU Load

```log
LLMManager: Using persistent OpenCL shader cache: /data/user/10/com.tcs.vehicleassistant/code_cache/litert_gpu_cache
litert    : [gpu_environment.h:155] Created LiteRT GpuEnvironment.
tflite    : Replacing 2068 out of 2068 node(s) with delegate (LITERT_CL) node, yielding 1 partitions for subgraph 0 (decode).
tflite    : Replacing 1107 out of 1107 node(s) with delegate (LITERT_CL) node, yielding 1 partitions for subgraph 1 (prefill_1024).
tflite    : Replacing 1107 out of 1107 node(s) with delegate (LITERT_CL) node, yielding 1 partitions for subgraph 2 (prefill_128).
tflite    : Replacing 2243 out of 2243 node(s) with delegate (LITERT_CL) node, yielding 1 partitions for subgraph 3 (verify).
LLMManager: Engine initialized with backend=GPU, maxNumTokens=4096
LLMManager: LLM Initialized successfully from /data/local/tmp/in_car_assistant_gemma4_e2b_pixel_int4.litertlm
LLMManager: Prewarm complete. KV cache populated.
```
