import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skipping full model initialization test locally if Cuda is missing. Otherwise it can be extremely slow/OOM.")
def test_tokenizer_and_model_init():
    """
    Tests if a typical tokenizer and model pipeline configures without breaking.
    Using a tiny model to prevent memory issues during unit testing.
    """
    # Use a tiny dummy model for basic structure validation instead of downloading >10GB weights
    model_id = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert tokenizer is not None, "Tokenizer failed to instantiate properly."
    
    model = AutoModelForCausalLM.from_pretrained(model_id)
    assert model is not None, "Model failed to instantiate properly."

def test_tokenizer_mock_inference():
    """Tests tokenization mapping."""
    model_id = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    text = "Sistem: Sen bir asistansın.\nSoru: Merhaba nasılsın?\nCevap:"
    tokens = tokenizer(text)
    
    assert "input_ids" in tokens
    assert "attention_mask" in tokens
    assert len(tokens["input_ids"]) > 0
