"""
Phase 1: Fine-Tuning Google Gemma 4-E2B with 4-bit QLoRA (Optimized for 8GB RTX 4070 VRAM)
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# Prevent PyTorch CUDA memory fragmentation on 8GB GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

SYSTEM_PROMPT_TEMPLATE = """CORE IDENTITY:
You are the driver's smart in-car AI Assistant and co-pilot. You help with vehicle controls (AC, temperature, windows, seat heaters, defrosters), navigation, music playback, phone calls, vehicle diagnostics, and travel suggestions. NEVER refer to yourself as a generic large language model or describe text processing algorithms.
PERSONALITY: Act as a warm, helpful, and direct in-car AI co-pilot.

User: {user_input}
Assistant: {output}"""

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4-E2B using 4-bit QLoRA on GPU")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-E2B-it", help="HuggingFace Base Model ID")
    parser.add_argument("--dataset_path", type=str, default="dataset/production_vehicle_dataset.json", help="Path to JSON dataset")
    parser.add_argument("--output_dir", type=str, default="in_car_gemma4_e2b_production_lora", help="Output directory for LoRA adapters")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()

def format_prompts(batch):
    formatted = []
    for inst, user_inp, out in zip(batch.get("instruction", [""]*len(batch["output"])), 
                                   batch.get("user", batch.get("input", [""]*len(batch["output"]))), 
                                   batch["output"]):
        u_text = user_inp if user_inp else inst
        text = f"<bos><start_of_turn>user\n{SYSTEM_PROMPT_TEMPLATE.format(user_input=u_text, output='')}<end_of_turn>\n<start_of_turn>model\n{out}<end_of_turn><eos>"
        formatted.append(text)
    return {"text": formatted}

def main():
    args = parse_args()
    print("=========================================================================")
    print(f"  PHASE 1: 4-BIT QLoRA GPU FINE-TUNING {args.model_id} ON {args.dataset_path}")
    print("=========================================================================\n")

    dataset = load_dataset("json", data_files=args.dataset_path)
    train_data = dataset["train"].shuffle(seed=42)

    formatted_dataset = train_data.map(format_prompts, batched=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    is_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    print(f"🚀 Hardware Acceleration: {'CUDA GPU (' + device_name + ')' if is_cuda else 'CPU Host'}")

    if is_cuda:
        torch.cuda.empty_cache()

    # 4-bit NormalFloat quantization config for ~4.8 GB total VRAM usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config if is_cuda else None,
        device_map={"": 0} if is_cuda else "cpu"
    )

    if is_cuda:
        model.gradient_checkpointing_enable()

    # Explicitly target language_model attention projection layers for Gemma 4 (5.35M params)
    target_modules = [
        f"language_model.layers.{i}.self_attn.{proj}"
        for i in range(35)
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
    ]

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch_fused" if is_cuda else "adamw_torch",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=is_cuda,
        use_cpu=not is_cuda,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=128,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    print(f"\nStarting 4-bit QLoRA GPU Fine-Tuning across {len(train_data):,} samples...")
    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Fine-tuning complete! LoRA adapters saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
