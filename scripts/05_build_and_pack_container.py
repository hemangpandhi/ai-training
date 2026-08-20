"""
Phase 5: Assembling model.toml, Extracting Official Baseline Assets (Tokenizer, Embedders, MTP Drafter), and Packing .litertlm Bundle
"""

import os
import shutil
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Assemble model.toml and pack LiteRT-LM container bundle")
    parser.add_argument("--bundle_dir", type=str, default="litert_export", help="Bundle directory containing prefill_decode TFLite model")
    parser.add_argument("--official_baseline_container", type=str, default=None, help="Optional path to official gemma-4-E2B-it.litertlm container to extract Tokenizer, Embedders, and MTP Drafter")
    parser.add_argument("--output_path", type=str, default="in_car_assistant_gemma4_e2b_pixel_int4.litertlm", help="Output .litertlm file")
    return parser.parse_args()

def extract_official_assets(official_container_path, bundle_dir):
    print(f"Unpacking official baseline container: {official_container_path}...")
    temp_dir = "/tmp/official_gemma4_baseline_unpacked"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        
    unpack_cmd = ["litert-lm", "unpack", official_container_path, "--output-dir", temp_dir]
    res = subprocess.run(unpack_cmd, capture_output=True, text=True)
    
    if res.returncode != 0:
        print("⚠️ Failed to unpack official container via litert-lm unpack:", res.stderr)
        return False

    # Key assets to copy from official baseline
    asset_mappings = {
        "Section1_SP_Tokenizer.spiece": "Section1_SP_Tokenizer.spiece",
        "Section2_TFLiteModel_tf_lite_embedder.tflite": "Section2_TFLiteModel_tf_lite_embedder.tflite",
        "Section3_TFLiteModel_tf_lite_per_layer_embedder.tflite": "Section3_TFLiteModel_tf_lite_per_layer_embedder.tflite",
        "Section11_TFLiteModel_tf_lite_mtp_drafter.tflite": "Section11_TFLiteModel_tf_lite_mtp_drafter.tflite",
        "LlmMetadataProto.pbtext": "LlmMetadataProto.pbtext"
    }

    print("Copying official Tokenizer, Single-Subgraph Embedders, and MTP Drafter into bundle...")
    for src_file, dst_file in asset_mappings.items():
        src_path = os.path.join(temp_dir, src_file)
        dst_path = os.path.join(bundle_dir, dst_file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            print(f"  ✅ Extracted and copied {src_file} -> {dst_path}")
        else:
            print(f"  ⚠️ Warning: {src_file} not found in official baseline.")

    return True

def generate_model_toml(output_dir):
    content = """[system_metadata]
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
"""
    with open(os.path.join(output_dir, 'model.toml'), 'w') as f:
        f.write(content)
    print("Generated model.toml manifest!")

def main():
    args = parse_args()
    print("=========================================================================")
    print(f"  PHASE 5: ASSEMBLING & PACKING CONTAINER TO {args.output_path}")
    print("=========================================================================\n")

    if args.official_baseline_container:
        extract_official_assets(args.official_baseline_container, args.bundle_dir)

    generate_model_toml(args.bundle_dir)

    cmd = [
        "litert-lm", "pack",
        args.bundle_dir,
        f"--output={args.output_path}",
        "--allow-overwrite"
    ]

    print("Executing command:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)

    print("STDOUT:\n", res.stdout)
    if res.stderr:
        print("STDERR:\n", res.stderr)

    if os.path.exists(args.output_path):
        size_mb = os.path.getsize(args.output_path) / (1024 * 1024)
        print(f"\n=========================================================================")
        print(f"  🎉 PACKAGING SUCCESS: {args.output_path} ({size_mb:.2f} MB)")
        print(f"=========================================================================")

if __name__ == "__main__":
    main()
