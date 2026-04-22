import argparse
import torch
import evaluate
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data_prep import load_and_prepare_dataset
from tqdm import tqdm

def generate_response(model, tokenizer, query: str, context: str = "", max_new_tokens: int = 256) -> str:
    """
    Generates an answer from the model given a prompt.
    """
    system_prompt = "Sen bir Türk hukuk asistanısın. Kullanıcının hukuki sorularını doğru ve eksiksiz bir şekilde yanıtla."
    if context and context.strip():
        prompt = f"Sistem: {system_prompt}\n\nSoru: {query}\n\nBağlam: {context}\n\nCevap:"
    else:
        prompt = f"Sistem: {system_prompt}\n\nSoru: {query}\n\nCevap:"
        
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )
    
    # Strip prompt from the beginning to return strictly the generation
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

def run_evaluation(model_name: str, adapter_path: str = None, sample_size: int = None, load_in_4bit: bool = False):
    """
    Evaluates a model (base or fine-tuned) on the shared test split.
    """
    dataset = load_and_prepare_dataset()
    test_data = dataset["test"]
    
    if sample_size and sample_size < len(test_data):
        test_data = test_data.select(range(sample_size))
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    kwargs = {"device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        kwargs["torch_dtype"] = torch.float16
        
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    
    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
    # Evaluate against exact string match (ROUGE is standard for Seq2Seq, works reasonably for causal LM QA as a metric proxy)
    rouge = evaluate.load("rouge")
    
    predictions = []
    references = []
    
    for item in tqdm(test_data, desc=f"Evaluating {model_name}"):
        query = item.get("instruction", item.get("question", item.get("soru", "")))
        context = item.get("input", item.get("context", ""))
        reference = item.get("output", item.get("answer", item.get("cevap", "")))
        
        pred = generate_response(model, tokenizer, query, context)
        predictions.append(pred)
        references.append(reference)
        
    results = rouge.compute(predictions=predictions, references=references)
    print("\n===============================")
    print(f"Evaluation Results for model: {model_name}")
    print(f"Adapter: {adapter_path if adapter_path else 'Base Model'}")
    print("===============================")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a baseline or finetuned model.")
    parser.add_argument("--model_name", type=str, required=True, help="Base HF model ID")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA weights (if fine-tuned)")
    parser.add_argument("--sample_size", type=int, default=None, help="Limit number of samples evaluated")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load reference model in 4-bit to prevent memory exhaustion")
    args = parser.parse_args()
    
    run_evaluation(args.model_name, args.adapter_path, args.sample_size, args.load_in_4bit)
