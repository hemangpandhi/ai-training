"""
Phase 2: Merging LoRA Adapter Weights back into Base Gemma 4-E2B Weights
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA Adapter weights into base Gemma 4-E2B model")
    parser.add_argument("--base_model", type=str, default="google/gemma-4-E2B-it", help="Base model identifier")
    parser.add_argument("--adapter_dir", type=str, default="in_car_gemma4_e2b_lora", help="Path to trained LoRA directory")
    parser.add_argument("--output_dir", type=str, default="gemma4_hf_merged", help="Output directory for merged model")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=========================================================================")
    print(f"  PHASE 2: MERGING LORA ADAPTER FROM {args.adapter_dir}")
    print("=========================================================================\n")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=True)

    print("Loading base model in FP16/BF16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )

    print("Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging weights into standalone model...")
    merged_model = model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving merged standalone model to {args.output_dir}...")
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✅ Merging complete! Merged HF checkpoint ready at: {args.output_dir}")

if __name__ == "__main__":
    main()
