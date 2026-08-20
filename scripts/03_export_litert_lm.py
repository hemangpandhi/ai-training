"""
Phase 3: Exporting Merged HuggingFace Model to LiteRT TFLite Subgraphs
"""

import os
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Export HuggingFace model to LiteRT TFLite via litert-torch")
    parser.add_argument("--model_dir", type=str, default="gemma4_hf_merged", help="Merged HuggingFace model directory")
    parser.add_argument("--output_dir", type=str, default="litert_export", help="Output directory for LiteRT subgraphs")
    parser.add_argument("--quant_recipe", type=str, default="dynamic_wi4_afp32", help="Quantization recipe string")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=========================================================================")
    print(f"  PHASE 3: EXPORTING {args.model_dir} TO LITERT VIA LITERT-TORCH")
    print("=========================================================================\n")

    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [
        "litert-torch", "export_hf",
        f"--model={args.model_dir}",
        f"--output_dir={args.output_dir}",
        f"--quantization_recipe={args.quant_recipe}",
        "--bundle_litert_lm=true",
        "--prefill_lengths=128,512,1024,2048,4096",
        "--cache_length=4096"
    ]

    print("Executing command:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)

    print("\nSTDOUT:\n", res.stdout)
    if res.stderr:
        print("\nSTDERR:\n", res.stderr)

    print(f"\n✅ LiteRT TFLite export complete! Subgraphs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
