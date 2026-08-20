"""
Optional Utility 7: Direct Weight Buffer Transposition from Fine-Tuned Model into Official Baseline Model
"""

import os
import struct
import argparse
import tflite

def parse_args():
    parser = argparse.ArgumentParser(description="Inject fine-tuned weight buffers into baseline TFLite model")
    parser.add_argument("--baseline_path", type=str, required=True, help="Path to official baseline prefill_decode TFLite model")
    parser.add_argument("--fine_tuned_path", type=str, required=True, help="Path to fine-tuned prefill_decode TFLite model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output aligned fine-tuned TFLite model")
    return parser.parse_args()

def inject_weights(baseline_path, fine_tuned_path, output_path):
    with open(baseline_path, 'rb') as f:
        off_buf = bytearray(f.read())

    with open(fine_tuned_path, 'rb') as f:
        fine_buf = bytearray(f.read())

    off_model = tflite.Model.GetRootAsModel(off_buf, 0)
    fine_model = tflite.Model.GetRootAsModel(fine_buf, 0)

    # Extract fine-tuned weight data map
    fine_weight_data = {}
    for i in range(fine_model.SubgraphsLength()):
        sub = fine_model.Subgraphs(i)
        for j in range(sub.TensorsLength()):
            t = sub.Tensors(j)
            name = t.Name().decode('utf-8') if t.Name() else ""
            b_idx = t.Buffer()
            if b_idx > 0 and name:
                buf_obj = fine_model.Buffers(b_idx)
                d_len = buf_obj.DataLength()
                if d_len > 0:
                    data_voff = buf_obj._tab.Offset(4)
                    if data_voff != 0:
                        data_ptr = buf_obj._tab.Pos + data_voff + struct.unpack_from('<I', fine_buf, buf_obj._tab.Pos + data_voff)[0]
                        raw_bytes = bytes(fine_buf[data_ptr+4 : data_ptr+4+d_len])
                        fine_weight_data[name] = raw_bytes

    print(f"Extracted {len(fine_weight_data)} fine-tuned weight tensors!")

    # Inject matching fine-tuned weights into baseline buffer
    injected_count = 0
    size_mismatch = 0

    for i in range(off_model.SubgraphsLength()):
        sub = off_model.Subgraphs(i)
        for j in range(sub.TensorsLength()):
            t = sub.Tensors(j)
            name = t.Name().decode('utf-8') if t.Name() else ""
            b_idx = t.Buffer()
            if b_idx > 0 and name in fine_weight_data:
                new_data = fine_weight_data[name]
                buf_obj = off_model.Buffers(b_idx)
                d_len = buf_obj.DataLength()
                if d_len == len(new_data):
                    data_voff = buf_obj._tab.Offset(4)
                    if data_voff != 0:
                        data_ptr = buf_obj._tab.Pos + data_voff + struct.unpack_from('<I', off_buf, buf_obj._tab.Pos + data_voff)[0]
                        off_buf[data_ptr+4 : data_ptr+4+d_len] = new_data
                        injected_count += 1
                else:
                    size_mismatch += 1

    print(f"Injection Complete: Successfully injected {injected_count} weight tensors! (Size mismatch={size_mismatch})")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(off_buf)

    print(f"Saved aligned fine-tuned TFLite model to {output_path}!")

def main():
    args = parse_args()
    print("=========================================================================")
    print("  UTILITY 7: DIRECT WEIGHT BUFFER TRANSPOSITION INTO BASELINE MODEL")
    print("=========================================================================\n")

    inject_weights(args.baseline_path, args.fine_tuned_path, args.output_path)

if __name__ == "__main__":
    main()
