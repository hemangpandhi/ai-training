"""
Phase 1: Fine-Tuning Google Gemma 4-E2B with PEFT LoRA
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

SYSTEM_PROMPT_TEMPLATE = """CORE IDENTITY:
You are the driver's smart in-car AI Assistant and co-pilot. You help with vehicle controls (AC, temperature, windows, seat heaters, defrosters), navigation, music playback, phone calls, vehicle diagnostics, and travel suggestions. NEVER refer to yourself as a generic large language model or describe text processing algorithms.
PERSONALITY: Act as a warm, helpful, and direct in-car AI co-pilot.

User: {user_input}
Assistant: {output}"""

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4-E2B using LoRA")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-E2B-it", help="HuggingFace Base Model ID")
    parser.add_argument("--dataset_path", type=str, default="dataset/production_vehicle_dataset.json", help="Path to JSON dataset")
    parser.add_argument("--output_dir", type=str, default="in_car_gemma4_e2b_production_lora", help="Output directory for LoRA adapters")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
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
    print(f"  PHASE 1: FINE-TUNING {args.model_id} ON {args.dataset_path}")
    print("=========================================================================\n")

    dataset = load_dataset("json", data_files=args.dataset_path)
    formatted_dataset = dataset["train"].map(format_prompts, batched=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Explicitly target language_model attention and MLP projection layers for Gemma 4
    target_modules = [
        f"language_model.layers.{i}.self_attn.{proj}"
        for i in range(35)
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
    ] + [
        f"language_model.layers.{i}.mlp.{proj}"
        for i in range(35)
        for proj in ["gate_proj", "up_proj", "down_proj"]
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

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("\nStarting LoRA Fine-Tuning...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Fine-tuning complete! LoRA adapters saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
