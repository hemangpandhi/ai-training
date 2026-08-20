# LiteRT Container Architecture & Section Specifications

This document outlines the container format, metadata requirements, and subgraph layout expected by the **LiteRT-LM C++ Engine** for Google Gemma 4-E2B models on Android.

---

## 🏗️ Container Layout (`.litertlm`)

A `.litertlm` file is a ZIP-like FlatBuffer bundle configured by a root `model.toml` manifest:

| Section ID | `model_type` | Data File | Description |
| :--- | :--- | :--- | :--- |
| **Section 0** | `LlmMetadata` | `LlmMetadataProto.pbtext` | Proto metadata containing prompt template, stop tokens, max tokens. |
| **Section 1** | `SP_Tokenizer` | `Section1_SP_Tokenizer.spiece` | SentencePiece tokenizer binary (`tokenizer.model`). |
| **Section 2** | `embedder` | `Section2_TFLiteModel_tf_lite_embedder.tflite` | Single-subgraph token embedder (`token_ids -> embeddings`). |
| **Section 3** | `per_layer_embedder` | `Section3_TFLiteModel_tf_lite_per_layer_embedder.tflite` | Single-subgraph per-layer token embedder. |
| **Section 10** | `prefill_decode` | `Section10_TFLiteModel_tf_lite_prefill_decode.tflite` | Main LLM prefill & decode transformer graph. |
| **Section 11** | `mtp_drafter` | `Section11_TFLiteModel_tf_lite_mtp_drafter.tflite` | Speculative decoding 3-token parallel drafter head. |

---

## 🧩 Section 10 Subgraph Array Ordering

The LiteRT-LM C++ runtime loader (`litert_compiled_model_executor_factory.cc`) enforces strict indexing for Subgraph 0:

- **Subgraph 0**: `decode` (single token decode graph, 31 outputs including `logits`)
- **Subgraph 1**: `prefill_4096` (max sequence length bucket)
- **Subgraph 2**: `prefill_2048`
- **Subgraph 3**: `prefill_1024`
- **Subgraph 4**: `prefill_512`
- **Subgraph 5**: `prefill_128` (min sequence length bucket)

---

## 🏷️ SignatureDefs Mapping

Each SignatureDef key in `tf_lite_prefill_decode.tflite` must map to its corresponding subgraph index:

- `decode` $\rightarrow$ Subgraph 0
- `prefill_4096` $\rightarrow$ Subgraph 1
- `prefill_2048` $\rightarrow$ Subgraph 2
- `prefill_1024` $\rightarrow$ Subgraph 3
- `prefill_512` $\rightarrow$ Subgraph 4
- `prefill_128` $\rightarrow$ Subgraph 5
