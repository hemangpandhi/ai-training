# `litert-torch` Executable Framework Patches

This document details the modifications made to the `litert-torch` site-package files to support Gemma 4-E2B LiteRT export and runtime compatibility.

---

## 🛠️ Patch 1: Tensor Name Canonicalization (`_convert/litert_converter.py`)

### Problem
During PyTorch `ExportedProgram` $\rightarrow$ MLIR $\rightarrow$ TFLite FlatBuffer conversion, `litert-torch` automatically prefixes input and output tensor names with sub-module names (e.g. `decode_kv_slice_0_output`, `prefill_128_embeddings`, etc.).
When the Android C++ `LiteRT-LM` loader parses the container, it throws:
```log
llm_litert_compiled_model_executor_factory.cc:197: Input tensor not found
```

### Solution
In `litert_torch/_convert/litert_converter.py`, we inject `_canonicalize_tensor_name()` inside `exported_programs_to_flatbuffer()`:

```python
def _canonicalize_tensor_name(name: str) -> str:
    import re

    clean = re.sub(
        r'^(decode|prefill_\d+|prefill_embedder_\d+|decode_embedder|embedder|prefill_per_layer_embedder_\d+|decode_per_layer_embedder|prefill_rope_\d+|decode_rope|prefill_mask_\d+|decode_mask)_',
        '',
        name,
    )
    clean = re.sub(r'_output$', '', clean)
    clean = re.sub(r'^kv_slice_', 'kv_cache_', clean)
    return clean


input_names = [_canonicalize_tensor_name(n) for n in raw_input_names]
output_names = [_canonicalize_tensor_name(n) for n in raw_output_names]
```

---

## 🛠️ Patch 2: JAX Tensor Shape Padding (`backend/lowerings/_jax_lowerings/lowerings.py`)

### Problem
During Gemma 4 RoPE and attention lowerings in JAX/MLIR, `torch.ops.aten.add.Tensor` fails when tensor ranks match but shape dimensions differ (e.g., broadcasting position embeddings across sequence length dimensions).

### Solution
In `litert_torch/backend/lowerings/_jax_lowerings/lowerings.py`, we update `_aten_add_tensor()` to handle shape padding:

```python
@registry.lower(torch.ops.aten.add.Tensor)
def _aten_add_tensor(lctx: LoweringContext, self, other):
    _log_usage(torch.ops.aten.add.Tensor)

    @jax_bridge.wrap
    def jax_lowering(self, other):
        promoted_type = jnp.promote_types(self.dtype, other.dtype)
        if promoted_type == jnp.float64:
            promoted_type = jnp.float32
        s_tensor = self.astype(promoted_type)
        o_tensor = other.astype(promoted_type)
        if s_tensor.shape != o_tensor.shape and len(s_tensor.shape) == len(
            o_tensor.shape
        ):
            max_shape = tuple(
                max(s, o) for s, o in zip(s_tensor.shape, o_tensor.shape)
            )
            if s_tensor.shape != max_shape:
                pad_widths = [
                    (0, m - s) for s, m in zip(s_tensor.shape, max_shape)
                ]
                s_tensor = jnp.pad(s_tensor, pad_widths)
            if o_tensor.shape != max_shape:
                pad_widths = [
                    (0, m - s) for s, m in zip(o_tensor.shape, max_shape)
                ]
                o_tensor = jnp.pad(o_tensor, pad_widths)
        return jnp.add(s_tensor, o_tensor)

    return jax_lowering(lctx, self, other)
```

---

## ⚡ Applying Patches

To apply these patches automatically to any Python environment:

```bash
python scripts/00_patch_litert_torch.py
```
