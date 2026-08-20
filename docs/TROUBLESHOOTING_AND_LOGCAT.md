# Complete Troubleshooting, Diagnostics & Logcat Reference

This document provides a comprehensive technical and executive diagnostic reference covering all runtime errors, root causes, FlatBuffer binary fixes, and resolution procedures encountered when training, exporting, and deploying **Google Gemma 4-E2B** models on Android Automotive OS and Pixel Tablets (`LITERT_CL` GPU Delegate).

---

## 🛠️ Issue Diagnostic Matrix

### Issue 1: `The model is not a valid Flatbuffer buffer` (`model_data.cc:437`)

- **🗣️ Layman Explanation**: The container failed to load on Android because the embedder section contained multiple messy subgraphs instead of a clean 1-subgraph structure.
- **🔬 Deep Technical Cause**: `Section 2` (`embedder`) or `Section 3` (`per_layer_embedder`) contained multi-bucket prefill subgraphs generated during naive `litert-torch` exports. LiteRT-LM C++ loader requires embedder models to have exactly 1 static lookup graph.
- **⚙️ Fix Procedure**: Use single-subgraph embedder models (`Section2_TFLiteModel_tf_lite_embedder.tflite`) extracted from the baseline container via `scripts/05_build_and_pack_container.py`.

---

### Issue 2: `Model requires one of [gpu_artisan] but Main backend is GPU`

- **🗣️ Layman Explanation**: Selecting `GPU` in the Android app crashed because the configuration file had a restrictive "gpu_artisan" rule hardcoded.
- **🔬 Deep Technical Cause**: `model.toml` contained `backend_constraint = "gpu_artisan"` under Section 10 (`prefill_decode`). When `LLMManager.kt` initializes with `Backend.GPU`, the runtime enforces exact constraint string matching and rejects standard OpenCL delegates.
- **⚙️ Fix Procedure**: Remove `backend_constraint` from `model.toml`. Use `prefer_activation_type = "fp16"` instead.

---

### Issue 3: `Input tensor not found` (`llm_litert_compiled_model_executor_factory.cc:197`)

- **🗣️ Layman Explanation**: Android couldn't find the model's single-token decode graph because it was hidden deeper in the file instead of being at the very beginning (Subgraph 0).
- **🔬 Deep Technical Cause**: `litert-torch` exporter saved `prefill_128` as Subgraph 0 and `decode` as Subgraph 5 in the FlatBuffer vector array. The LiteRT C++ loader hardcodes Subgraph 0 as `decode`.
- **⚙️ Fix Procedure**: Run `scripts/04_fix_subgraphs_and_signatures.py` to re-order the FlatBuffer `Subgraphs` vector array so Subgraph 0 is `decode`, followed by `prefill_4096` through `prefill_128`.

---

### Issue 4: `Signature has incorrect number of input/outputs` (`model_load.cc:396`)

- **🗣️ Layman Explanation**: The model's signature map pointed the "decode" label to a prefill graph, causing output shape mismatch (30 outputs instead of 31 outputs with `logits`).
- **🔬 Deep Technical Cause**: `SignatureDef` table array index offsets were misaligned, pointing key `'decode'` to Subgraph 5 (`prefill_128`).
- **⚙️ Fix Procedure**: Run `scripts/04_fix_subgraphs_and_signatures.py` to align SignatureDef key `'decode'` to Subgraph 0 and sequence bucket keys to their respective prefill subgraphs.

---

### Issue 5: SignatureDef `subgraph_index` vtable offset fallback to 0

- **🗣️ Layman Explanation**: A bug in FlatBuffer binary serialization caused all signature keys to point to Subgraph 0.
- **🔬 Deep Technical Cause**: FlatBuffer omits zero values in vtable field offset tables. When `subgraph_index = 0`, the field offset is omitted (vtable entry = 0), causing runtime getters to read default 0 for all keys.
- **⚙️ Fix Procedure**: Explicitly construct vtable entry offsets in `scripts/04_fix_subgraphs_and_signatures.py` to preserve signature-to-subgraph mapping.

---

### Issue 6: Unprefixed vs Prefixed Tensor Names (`decode_kv_slice_0` vs `kv_cache_0`)

- **🗣️ Layman Explanation**: Exported tensors had prefix names attached (like `decode_kv_slice_0`), so the C++ engine couldn't match them to expected runtime state names (`kv_cache_0`).
- **🔬 Deep Technical Cause**: PyTorch `ExportedProgram` sub-module namespaces leak into tensor names during MLIR conversion.
- **⚙️ Fix Procedure**: Run `scripts/00_patch_litert_torch.py` which injects `_canonicalize_tensor_name()` into `litert_converter.py` to strip module prefixes.

---

### Issue 7: JAX Tensor Shape Padding Mismatch (`_aten_add_tensor`)

- **🗣️ Layman Explanation**: Position embedding additions failed during MLIR lowerings due to shape size mismatches across layers.
- **🔬 Deep Technical Cause**: Gemma 4 heterogeneous attention layers broadcast tensors of matching rank but mismatched dimension sizes during `torch.ops.aten.add.Tensor`.
- **⚙️ Fix Procedure**: Run `scripts/00_patch_litert_torch.py` to inject dynamic shape padding (`jnp.pad`) into `lowerings.py`.

---

### Issue 8: Heterogeneous Layer & Static Weight Mismatch

- **🗣️ Layman Explanation**: Gemma 4 contains different layer types (multimodal attention vs per-layer embeddings). Static weights were conflicting with dynamic prompt sequence buckets.
- **🔬 Deep Technical Cause**: Gemma 4 alternates full global attention with local sliding-window attention and per-layer embedding projections. Static weight matrices were getting re-quantized or duplicated per sequence length bucket.
- **⚙️ Fix Procedure**: Use unified static weight buffer IDs in FlatBuffer subgraphs via canonicalization, and use `scripts/07_inject_weights_into_baseline.py` to transpose fine-tuned weight buffers directly into baseline model positions.

---

### Issue 9: Speculative Decoding MTP Drafter Alignment (`Section 11`)

- **🗣️ Layman Explanation**: Generation throughput was stuck at ~7 TPS because speculative decoding was inactive.
- **🔬 Deep Technical Cause**: Missing `Section11_TFLiteModel_tf_lite_mtp_drafter.tflite` (30-subgraph drafter head) and missing `model_type = "mtp_drafter"` section in `model.toml`.
- **⚙️ Fix Procedure**: Use `scripts/05_build_and_pack_container.py` to extract the official 3-token MTP drafter head and pack it into the container bundle, increasing speed to **22–25 TPS**.

---

### Issue 10: Android UI Automation Center-Click Selection

- **🗣️ Layman Explanation**: ADB commands failed to tap the right model or GPU backend because screen coordinates were hardcoded.
- **🔬 Deep Technical Cause**: Screen resolutions and pixel density vary across Android Automotive OS displays and Pixel Tablets.
- **⚙️ Fix Procedure**: Run `scripts/06_android_ui_automation.py` to dump `uiautomator dump` XML, parse exact node bounds `[x1,y1][x2,y2]`, and calculate dynamic center coordinates `((x1+x2)/2, (y1+y2)/2)`.

---

## 🔍 Real-Time Logcat Verification Stream

To monitor device execution on Android Automotive OS or Pixel Tablet:

```bash
adb -s 3704105H8094TU shell "logcat -v time LLMManager:V LiteRT:V litert_lm:V tflite:V *:S"
```

### ✅ Expected Logcat Output for Successful GPU Initialization

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
