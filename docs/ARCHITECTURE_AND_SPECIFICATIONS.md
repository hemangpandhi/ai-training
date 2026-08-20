# LiteRT Container Architecture, Section Manifest & FlatBuffer Specifications

This document provides a comprehensive technical reference for the `.litertlm` container bundle format, section declarations in `model.toml`, FlatBuffer binary offsets, and OpenCL GPU delegate rules for **Google Gemma 4-E2B** on Android.

---

## 📦 Container Section Manifest (`model.toml`)

The root `model.toml` configures how LiteRT-LM C++ runtime (`litert_lm_loader.cc`) maps container data streams to model components:

```toml
[system_metadata]
entries = [
  { key = "author", value_type = "String", value = "Google ODML / In-Car AI Team" },
  { key = "uuid", value_type = "String", value = "2fa073f5-2d5e-44ff-8bb9-64d926dc40e2" },
  { key = "creation_timestamp", value_type = "String", value = "2026-08-20T11:00:00.000000+00:00" },
]

[[section]]
section_type = "LlmMetadata"
data_path = "LlmMetadataProto.pbtext"

[[section]]
section_type = "SP_Tokenizer"
data_path = "Section1_SP_Tokenizer.spiece"

[[section]]
model_type = "embedder"
section_type = "TFLiteModel"
data_path = "Section2_TFLiteModel_tf_lite_embedder.tflite"

[[section]]
model_type = "per_layer_embedder"
section_type = "TFLiteModel"
data_path = "Section3_TFLiteModel_tf_lite_per_layer_embedder.tflite"

[[section]]
additional_metadata = [
  { key = "prefer_activation_type", value_type = "String", value = "fp16" },
]
model_type = "prefill_decode"
section_type = "TFLiteModel"
data_path = "Section10_TFLiteModel_tf_lite_prefill_decode.tflite"

[[section]]
model_type = "mtp_drafter"
section_type = "TFLiteModel"
data_path = "Section11_TFLiteModel_tf_lite_mtp_drafter.tflite"
```

### ⚠️ Critical Manifest Rules
1. **Section Key `model_type`**: Must match `"prefill_decode"` for Section 10. `LLMManager.kt` explicitly checks for container section key `"prefill_decode"`.
2. **Backend Constraints**: **DO NOT** add `backend_constraint = "gpu_artisan"` unless exporting a specialized artisan binary. Adding `gpu_artisan` constraint causes standard OpenCL GPU initialization to fail with `Model requires one of [gpu_artisan] but Main backend is GPU`.
3. **Embedder Sections**: `Section 2` (`embedder`) and `Section 3` (`per_layer_embedder`) must contain **EXACTLY 1 SUBGRAPH** (`Subgraph 0: main`). Multi-bucket embedder subgraphs cause `Signature has incorrect number of input/outputs` during C++ model loading.

---

## 🧩 Section 10 (`tf_lite_prefill_decode.tflite`) Subgraph Ordering

The LiteRT-LM C++ executor factory (`llm_litert_compiled_model_executor_factory.cc`) expects strict subgraph array ordering:

- **Subgraph 0**: `decode` (single token decode graph, 31 outputs including `logits`)
- **Subgraph 1**: `prefill_4096` (max sequence length bucket)
- **Subgraph 2**: `prefill_2048`
- **Subgraph 3**: `prefill_1024`
- **Subgraph 4**: `prefill_512`
- **Subgraph 5**: `prefill_128` (min sequence length bucket)

### Why Order Matters
If Subgraph 0 is `prefill_128` instead of `decode`, LiteRT-LM looks for input tensors of `decode` inside `prefill_128` and fails at line 197:
```log
llm_litert_compiled_model_executor_factory.cc:197: Input tensor not found
```

---

## 🏷️ SignatureDefs & FlatBuffer Table Vtable Offsets

In TFLite FlatBuffer binary schema (`SignatureDef` table):
- **Field 0**: `inputs` (vtable offset 4)
- **Field 1**: `outputs` (vtable offset 6)
- **Field 2**: `signature_key` (vtable offset 8)
- **Field 3**: `subgraph_index` (vtable offset 10)

### SignatureDef Mapping
Each SignatureDef key must map to its exact target subgraph:
- `key='decode'` $\rightarrow$ `subgraph_index=0` (`decode`)
- `key='prefill_4096'` $\rightarrow$ `subgraph_index=1` (`prefill_4096`)
- `key='prefill_2048'` $\rightarrow$ `subgraph_index=2` (`prefill_2048`)
- `key='prefill_1024'` $\rightarrow$ `subgraph_index=3` (`prefill_1024`)
- `key='prefill_512'` $\rightarrow$ `subgraph_index=4` (`prefill_512`)
- `key='prefill_128'` $\rightarrow$ `subgraph_index=5` (`prefill_128`)

---

## 🚀 Speculative Decoding (`mtp_drafter`) Section

Section 11 contains `Section11_TFLiteModel_tf_lite_mtp_drafter.tflite`:
- **Subgraphs count**: 30
- **SignatureDef 0**: `key='mtp_drafter' -> Subgraph 0`
- **Function**: Enables 3-token parallel candidate generation per OpenCL forward step, accelerating TPS from 7.5 → 22–25 TPS.
