from typing import Optional, List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class HuggingFaceLLM:
    """A simple wrapper for Hugging Face language models."""
    
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None):
        """Initialize the Hugging Face language model.
        
        Args:
            model_name: Name of the Hugging Face model to use
            device: Device to run the model on (e.g., 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load the model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            logger.info(f"Loading model {self.model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if 'cuda' in self.device else torch.float32,
                device_map="auto" if 'cuda' in self.device else None
            ).to(self.device)
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from the model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            self.load()
            
        try:
            # Encode the input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            input_length = inputs["input_ids"].shape[1]
            
            # Set max_new_tokens to a reasonable value (e.g., 200) or based on model's max length
            max_new_tokens = min(
                200,  # Default max new tokens
                self.model.config.max_position_embeddings - input_length - 1
            )
            
            if max_new_tokens <= 0:
                return "[Error: Input is too long. Please try a shorter query or split your input into smaller chunks.]"
            
            # Generate the output
            outputs = self.model.generate(
                input_ids=inputs["input_ids"].to(self.device),
                attention_mask=inputs["attention_mask"].to(self.device),
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Decode the output
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input prompt from the output if it's there
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()
                
            return generated.strip()
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}", exc_info=True)
            return f"Error generating response: {str(e)}"

# Create a default instance
default_llm = HuggingFaceLLM()
