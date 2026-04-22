import pytest
from src.data_prep import load_and_prepare_dataset, apply_prompt_template

def test_dataset_loading():
    """Tests if the dataset is loaded properly with correct splits."""
    dataset = load_and_prepare_dataset("Renicames/turkish-law-chatbot")
    
    assert "train" in dataset, "Dataset does not contain a 'train' split."
    assert "test" in dataset, "Dataset does not contain a 'test' split."
    assert len(dataset["train"]) > 0, "'train' split is empty."
    assert len(dataset["test"]) > 0, "'test' split is empty."

def test_prompt_formatting():
    """Tests if the prompt formatting maps fields correctly and creates the text column."""
    dataset = load_and_prepare_dataset("Renicames/turkish-law-chatbot")
    
    # Selecting a tiny subset to make test fast
    dataset["train"] = dataset["train"].select(range(5))
    
    formatted_dataset = apply_prompt_template(dataset)
    assert "text" in formatted_dataset["train"].column_names, "'text' column not found after formatting."
    
    sample_text = formatted_dataset["train"][0]["text"]
    assert isinstance(sample_text, str), "The generated prompt should be a string."
    assert len(sample_text.strip()) > 0, "The generated prompt string is empty."
    assert "Sistem:" in sample_text, "System prompt seems to be missing."
