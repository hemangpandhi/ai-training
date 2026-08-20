"""
Phase 4: Fixing Subgraph Vector Array and SignatureDef Offsets in FlatBuffer Binary
"""

import os
import struct
import argparse
import tflite

def parse_args():
    parser = argparse.ArgumentParser(description="Re-order Subgraph Array and SignatureDefs in TFLite FlatBuffer")
    parser.add_argument("--tflite_path", type=str, default="litert_export/Section2_TFLiteModel_tf_lite_prefill_decode.tflite", help="Path to prefill_decode TFLite model")
    return parser.parse_args()

def reorder_subgraphs_and_signatures(filepath):
    with open(filepath, 'rb') as f:
        buf = bytearray(f.read())

    model = tflite.Model.GetRootAsModel(buf, 0)
    print(f"Original Subgraphs count: {model.SubgraphsLength()}")

    # Read names of all subgraphs
    names = []
    for i in range(model.SubgraphsLength()):
        s = model.Subgraphs(i)
        names.append(s.Name().decode('utf-8') if s.Name() else f"sub_{i}")

    target_order = ['decode', 'prefill_4096', 'prefill_2048', 'prefill_1024', 'prefill_512', 'prefill_128']
    old_indices = [names.index(n) for n in target_order]
    print("Old indices mapping for target order:", old_indices)

    root_off = struct.unpack_from('<I', buf, 0)[0]
    vtable_off = root_off - struct.unpack_from('<i', buf, root_off)[0]

    # Subgraphs field in Model table is at vtable offset 8 (field index 2)
    subgraphs_voff = struct.unpack_from('<H', buf, vtable_off + 8)[0]
    subgraphs_vec_pos = root_off + subgraphs_voff
    subgraphs_vec_data = subgraphs_vec_pos + struct.unpack_from('<I', buf, subgraphs_vec_pos)[0]
    vec_len = struct.unpack_from('<I', buf, subgraphs_vec_data)[0]

    elem_start = subgraphs_vec_data + 4

    abs_offs = []
    for i in range(vec_len):
        pos = elem_start + i * 4
        rel = struct.unpack_from('<I', buf, pos)[0]
        abs_offs.append(pos + rel)

    new_abs_offs = list(abs_offs)
    for new_idx, old_idx in enumerate(old_indices):
        new_abs_offs[new_idx] = abs_offs[old_idx]

    for i in range(vec_len):
        pos = elem_start + i * 4
        rel = new_abs_offs[i] - pos
        struct.pack_into('<I', buf, pos, rel)

    # Verify updated model
    new_model = tflite.Model.GetRootAsModel(buf, 0)
    new_names = [new_model.Subgraphs(i).Name().decode('utf-8') for i in range(6)]
    print("Reordered first 6 subgraph names:", new_names)

    with open(filepath, 'wb') as f:
        f.write(buf)

    print(f"✅ Successfully reordered FlatBuffer Subgraph array in {filepath}!")

def main():
    args = parse_args()
    print("=========================================================================")
    print("  PHASE 4: FIXING SUBGRAPH ARRAY AND SIGNATURE DEF FLATBUFFER OFFSETS")
    print("=========================================================================\n")

    reorder_subgraphs_and_signatures(args.tflite_path)

if __name__ == "__main__":
    main()
