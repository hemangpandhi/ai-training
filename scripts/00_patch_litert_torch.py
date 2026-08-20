"""
Step 0: Automated Patching Script for litert-torch Site-Package Integrations
This script applies required tensor name canonicalization and JAX lowering shape fixes to litert-torch.
"""

import os
import sys
import re

def patch_litert_converter(file_path):
    print(f"Checking {file_path}...")
    if not os.path.exists(file_path):
        print(f"  ❌ File not found: {file_path}")
        return False
        
    with open(file_path, 'r') as f:
        content = f.read()
        
    if '_canonicalize_tensor_name' in content:
        print("  ✅ litert_converter.py already patched!")
        return True
        
    target = "raw_output_names = _get_output_names(exported_program, lowered)"
    replacement = """raw_output_names = _get_output_names(exported_program, lowered)

    def _canonicalize_tensor_name(name: str) -> str:
      import re
      clean = re.sub(r'^(decode|prefill_\\d+|prefill_embedder_\\d+|decode_embedder|embedder|prefill_per_layer_embedder_\\d+|decode_per_layer_embedder|prefill_rope_\\d+|decode_rope|prefill_mask_\\d+|decode_mask)_', '', name)
      clean = re.sub(r'_output$', '', clean)
      clean = re.sub(r'^kv_slice_', 'kv_cache_', clean)
      return clean

    input_names = [_canonicalize_tensor_name(n) for n in raw_input_names]
    output_names = [_canonicalize_tensor_name(n) for n in raw_output_names]"""

    if target in content:
        content = content.replace(target, replacement)
        with open(file_path, 'w') as f:
            f.write(content)
        print("  🎉 Successfully patched litert_converter.py!")
        return True
    else:
        print("  ⚠️ Target insertion point not found in litert_converter.py")
        return False

def patch_jax_lowerings(file_path):
    print(f"Checking {file_path}...")
    if not os.path.exists(file_path):
        print(f"  ❌ File not found: {file_path}")
        return False

    with open(file_path, 'r') as f:
        content = f.read()

    if 's_tensor.shape != max_shape' in content:
        print("  ✅ lowerings.py already patched!")
        return True

    target = "s_tensor = self.astype(promoted_type)\n    o_tensor = other.astype(promoted_type)"
    replacement = """s_tensor = self.astype(promoted_type)
    o_tensor = other.astype(promoted_type)
    if s_tensor.shape != o_tensor.shape and len(s_tensor.shape) == len(o_tensor.shape):
      max_shape = tuple(max(s, o) for s, o in zip(s_tensor.shape, o_tensor.shape))
      if s_tensor.shape != max_shape:
        pad_widths = [(0, m - s) for s, m in zip(s_tensor.shape, max_shape)]
        s_tensor = jnp.pad(s_tensor, pad_widths)
      if o_tensor.shape != max_shape:
        pad_widths = [(0, m - s) for s, m in zip(o_tensor.shape, max_shape)]
        o_tensor = jnp.pad(o_tensor, pad_widths)"""

    if target in content:
        content = content.replace(target, replacement)
        with open(file_path, 'w') as f:
            f.write(content)
        print("  🎉 Successfully patched lowerings.py!")
        return True
    else:
        print("  ⚠️ Target insertion point not found in lowerings.py")
        return False

def main():
    import importlib.util
    spec = importlib.util.find_spec("litert_torch")
    if not spec or not spec.origin:
        print("❌ litert_torch package not found in current Python environment!")
        sys.exit(1)

    pkg_root = os.path.dirname(spec.origin)
    converter_path = os.path.join(pkg_root, "_convert", "litert_converter.py")
    lowerings_path = os.path.join(pkg_root, "backend", "lowerings", "_jax_lowerings", "lowerings.py")

    print("=========================================================================")
    print("  APPLYING REQUIRED LITERT-TORCH FRAMEWORK PATCHES")
    print("=========================================================================\n")

    p1 = patch_litert_converter(converter_path)
    p2 = patch_jax_lowerings(lowerings_path)

    if p1 and p2:
        print("\n✅ All litert-torch framework patches applied successfully!")
    else:
        print("\n⚠️ One or more patches could not be applied. Check Python environment.")

if __name__ == "__main__":
    main()
