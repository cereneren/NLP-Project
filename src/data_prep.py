import os
from typing import Dict, Any
from datasets import load_dataset, DatasetDict

def load_and_prepare_dataset(dataset_name: str = "Renicames/turkish-law-chatbot") -> DatasetDict:
    """
    Loads the dataset from Hugging Face and prepares it for training and evaluation.
    
    Args:
        dataset_name: The name of the dataset repository on Hugging Face.
        
    Returns:
        A DatasetDict containing 'train' and 'test' splits.
    """
    dataset = load_dataset(dataset_name)
    return dataset

def format_prompt(sample: Dict[str, Any]) -> str:
    """
    Formats the sample into a prompt suitable for instruction fine-tuning.
    
    Args:
        sample: A dictionary containing data fields.
        
    Returns:
        A formatted string.
    """
    system_prompt = "Sen bir Türk hukuk asistanısın. Kullanıcının hukuki sorularını doğru ve eksiksiz bir şekilde yanıtla."
    
    # Adapt to different dataset architectures dynamically
    instruction = sample.get("instruction", sample.get("question", sample.get("soru", "")))
    input_text = sample.get("input", sample.get("context", ""))
    output = sample.get("output", sample.get("answer", sample.get("cevap", "")))
    
    if input_text and str(input_text).strip():
        prompt = f"Sistem: {system_prompt}\n\nSoru: {instruction}\n\nBağlam: {input_text}\n\nCevap: {output}"
    else:
        prompt = f"Sistem: {system_prompt}\n\nSoru: {instruction}\n\nCevap: {output}"
        
    return prompt

def apply_prompt_template(dataset: DatasetDict) -> DatasetDict:
    """
    Applies the prompt formatting to all splits in the dataset.
    
    Args:
        dataset: The DatasetDict to format.
        
    Returns:
        A DatasetDict with an additional 'text' column containing the formatted prompts.
    """
    def _map_fn(example):
        return {"text": format_prompt(example)}
    
    return dataset.map(_map_fn)
