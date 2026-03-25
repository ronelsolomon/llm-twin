#!/usr/bin/env python3
"""
Simple inference script for the DPO model with better prompting
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SimpleDPOInference:
    def __init__(self, model_path="./dpo_llm_twin_merged"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model and tokenizer"""
        print(f"Loading model from: {self.model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ Model loaded successfully!")
        
    def generate_response(self, prompt, max_new_tokens=150):
        """Generate response with better prompting"""
        
        # Use conversational context instead of instruction format
        conversational_prompt = f"As Ronel Solomon, {prompt}"
        
        inputs = self.tokenizer(conversational_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        if conversational_prompt in response:
            response = response.replace(conversational_prompt, "").strip()
        
        return response
    
    def chat(self):
        """Simple chat interface"""
        self.load_model()
        
        print("\n💬 Chat with Ronel's AI Twin!")
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
                
            try:
                response = self.generate_response(user_input)
                print(f"Ronel AI: {response}\n")
            except Exception as e:
                print(f"Error: {e}\n")

if __name__ == "__main__":
    inference = SimpleDPOInference()
    inference.chat()
