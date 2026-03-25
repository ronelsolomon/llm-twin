#!/usr/bin/env python3
"""
Advanced Speculative Decoding Implementation for LLM Evaluation
"""

import torch
import time
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeculativeDecoder:
    """Advanced speculative decoding with draft models"""
    
    def __init__(self, main_model, tokenizer, device="cpu"):
        self.main_model = main_model
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize draft model (smaller, faster model)
        self.draft_model = None
        self.draft_tokenizer = None
        
        # Speculative decoding parameters
        self.speculative_steps = 4  # Number of tokens to speculate ahead
        self.acceptance_threshold = 0.8  # Minimum acceptance rate
        
    def load_draft_model(self, draft_model_name: str = "gpt2"):
        """Load a smaller draft model for speculation"""
        try:
            logger.info(f"Loading draft model: {draft_model_name}")
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_name,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == "cuda" else None
            ).to(self.device)
            
            self.draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
            if self.draft_tokenizer.pad_token is None:
                self.draft_tokenizer.pad_token = self.draft_tokenizer.eos_token
                
            logger.info(f"Draft model loaded successfully: {draft_model_name}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load draft model {draft_model_name}: {e}")
            return False
    
    def speculative_generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                         max_new_tokens: int = 512, temperature: float = 0.7) -> torch.Tensor:
        """Generate using speculative decoding"""
        
        if self.draft_model is None:
            # Fallback to regular generation
            return self._regular_generate(input_ids, attention_mask, max_new_tokens, temperature)
        
        try:
            return self._speculative_generate_internal(input_ids, attention_mask, max_new_tokens, temperature)
        except Exception as e:
            logger.warning(f"Speculative generation failed: {e}, falling back to regular generation")
            return self._regular_generate(input_ids, attention_mask, max_new_tokens, temperature)
    
    def _speculative_generate_internal(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                                  max_new_tokens: int, temperature: float) -> torch.Tensor:
        """Internal speculative generation logic"""
        
        generated_ids = input_ids.clone()
        total_generated = 0
        
        while total_generated < max_new_tokens:
            # Step 1: Draft model generates speculative tokens
            with torch.no_grad():
                draft_outputs = self.draft_model.generate(
                    generated_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=min(self.speculative_steps, max_new_tokens - total_generated),
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.draft_tokenizer.pad_token_id
                )
            
            draft_tokens = draft_outputs[0, generated_ids.shape[1]:]
            
            if len(draft_tokens) == 0:
                break
            
            # Step 2: Main model verifies and accepts/rejects tokens
            with torch.no_grad():
                main_outputs = self.main_model(
                    generated_ids,
                    attention_mask=attention_mask
                )
            
            # Step 3: Accept/reject logic
            accepted_tokens = self._verify_speculative_tokens(
                generated_ids, draft_tokens, main_outputs.logits, temperature
            )
            
            # Step 4: Update generated sequence
            if len(accepted_tokens) > 0:
                accepted_tensor = torch.tensor([accepted_tokens], device=self.device).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, accepted_tensor], dim=1)
                
                # Update attention mask
                new_attention = torch.ones_like(accepted_tensor)
                attention_mask = torch.cat([attention_mask, new_attention], dim=1)
                
                total_generated += len(accepted_tokens)
            else:
                # No tokens accepted, break
                break
            
            # Early stopping if we've generated enough
            if total_generated >= max_new_tokens:
                break
        
        return generated_ids
    
    def _verify_speculative_tokens(self, context_ids: torch.Tensor, draft_tokens: torch.Tensor,
                               main_logits: torch.Tensor, temperature: float) -> list:
        """Verify speculative tokens against main model predictions"""
        
        accepted_tokens = []
        
        for i, draft_token in enumerate(draft_tokens):
            if i >= main_logits.shape[1] - 1:
                break
                
            # Get main model's prediction for next token
            next_token_logits = main_logits[0, context_ids.shape[1] + i - 1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Get probabilities
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Check if draft token is in top-k predictions
            top_k = 5
            top_tokens = torch.topk(probs, top_k).indices.tolist()
            
            if draft_token.item() in top_tokens:
                # Calculate acceptance probability
                acceptance_prob = probs[draft_token].item()
                if acceptance_prob >= self.acceptance_threshold:
                    accepted_tokens.append(draft_token.item())
                else:
                    # Accept with probability
                    if torch.rand(1).item() < acceptance_prob:
                        accepted_tokens.append(draft_token.item())
                    else:
                        break
            else:
                break
        
        return accepted_tokens
    
    def _regular_generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       max_new_tokens: int, temperature: float) -> torch.Tensor:
        """Regular generation fallback"""
        
        with torch.no_grad():
            outputs = self.main_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        return outputs
    
    def benchmark_generation_speed(self, test_input: str, max_tokens: int = 100) -> Dict[str, float]:
        """Benchmark speculative vs regular generation"""
        
        inputs = self.tokenizer.encode(test_input, return_tensors="pt").to(self.device)
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Test regular generation
        start_time = time.time()
        regular_output = self._regular_generate(inputs, attention_mask, max_tokens, 0.7)
        regular_time = time.time() - start_time
        
        # Test speculative generation
        if self.draft_model:
            start_time = time.time()
            speculative_output = self.speculative_generate(inputs, attention_mask, max_tokens, 0.7)
            speculative_time = time.time() - start_time
            
            speedup = regular_time / speculative_time if speculative_time > 0 else 1.0
            
            return {
                "regular_time": regular_time,
                "speculative_time": speculative_time,
                "speedup": speedup,
                "regular_length": len(regular_output[0]),
                "speculative_length": len(speculative_output[0])
            }
        else:
            return {
                "regular_time": regular_time,
                "speculative_time": None,
                "speedup": 1.0,
                "regular_length": len(regular_output[0]),
                "speculative_length": None
            }

def create_speculative_decoder(model, tokenizer, device="cpu", draft_model="gpt2"):
    """Factory function to create speculative decoder"""
    
    decoder = SpeculativeDecoder(model, tokenizer, device)
    decoder.load_draft_model(draft_model)
    return decoder

# Usage example and testing
if __name__ == "__main__":
    # Test speculative decoding implementation
    print("=== Speculative Decoding Test ===")
    
    # This would be used in the model comparison evaluator
    test_input = "Explain the benefits of renewable energy in simple terms."
    
    print(f"Test input: {test_input}")
    print("To use speculative decoding:")
    print("1. Create SpeculativeDecoder with your model")
    print("2. Load a draft model (e.g., gpt2)")
    print("3. Use speculative_generate() instead of regular generate()")
    print("\nExpected speedup: 1.2x - 2.0x depending on model and hardware")
