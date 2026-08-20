"""
Phase 5: Assembling model.toml, SentencePiece Tokenizers, MTP Drafter, and Packing .litertlm Bundle
"""

import os
import shutil
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Assemble model.toml and pack LiteRT-LM container bundle")
    parser.add_argument("--bundle_dir", type=str, default="litert_export", help="Bundle directory")
    parser.add_argument("--output_path", type=str, default="in_car_assistant_gemma4_e2b_pixel_int4.litertlm", help="Output .litertlm file")
    return parser.parse_args()

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
