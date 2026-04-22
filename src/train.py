import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from src.data_prep import load_and_prepare_dataset, apply_prompt_template

def train(model_name: str, output_dir: str, num_train_epochs: int = 3, batch_size: int = 4, lr: float = 2e-4):
    """
    Fine-tunes the base model using QLoRA.
    
    Args:
        model_name: The Hugging Face model identifier to fine-tune.
        output_dir: The directory where the fine-tuned model and checkpoints will be saved.
        num_train_epochs: Number of times to iterate over the training dataset.
        batch_size: The batch size per device during training.
        lr: Learning rate for the optimizer.
    """
    dataset = load_and_prepare_dataset()
    dataset = apply_prompt_template(dataset)
    train_data = dataset["train"]
    
    # Configure 4-bit quantization using bitsandbytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA/QLoRA adapter
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        num_train_epochs=num_train_epochs,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        do_eval=True
    )
    
    # Ensure test set memory limits check out for eval if extremely large
    eval_data = dataset["test"]
    
    # Initialize Supervised Fine-Tuning Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args
    )
    
    # Execute training loop
    trainer.train()
    
    # Persist the final adapter weights and tokenizer configuration
    final_model_path = f"{output_dir}/final_model"
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an LLM using QLoRA.")
    parser.add_argument("--model_name", type=str, required=True, help="HF model ID to fine-tune")
    parser.add_argument("--output_dir", type=str, default="models/fine_tuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()
    
    train(args.model_name, args.output_dir, args.epochs, args.batch_size, args.lr)
