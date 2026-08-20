"""
Phase 1: Gemma 4-E2B Fine-Tuning Script using LoRA (PEFT / Unsloth / HuggingFace)
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4-E2B with LoRA for In-Car Assistant Tasks")
    parser.add_argument("--base_model", type=str, default="google/gemma-4-E2B-it", help="Base model checkpoint")
    parser.add_argument("--dataset_path", type=str, default="dataset/sample_in_car_dataset.json", help="Path to JSON dataset")
    parser.add_argument("--output_dir", type=str, default="in_car_gemma4_e2b_lora", help="Output directory for LoRA weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=========================================================================")
    print(f"  PHASE 1: FINE-TUNING {args.base_model} WITH LORA")
    print("=========================================================================\n")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=args.dataset_path)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        fp16=False,
        bf16=True,
        save_strategy="epoch",
        report_to="none"
    )

    def formatting_prompts_func(example):
        text = f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n<start_of_turn>model\n{example['output']}<end_of_turn>"
        return [text]

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=lora_config,
        dataset_text_field="instruction",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Fine-tuning complete! LoRA weights saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
