#!/usr/bin/env python3
"""
Script to use the DPO fine-tuned LLM twin model for inference
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import argparse

class DPOModelInference:
    def __init__(self, model_path: str = "./dpo_llm_twin_merged"):
        """Initialize the DPO model for inference"""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print(f"🤖 Loading model from: {self.model_path}")
        
        # Check if model exists
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        # Load model with appropriate device mapping
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"📱 Using device: {device}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto" if device != "cpu" else None
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ Model and tokenizer loaded successfully!")
        
    def generate_response(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, do_sample: bool = True):
        """Generate a response from the model"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Format prompt using Alpaca template (consistent with training)
        alpaca_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:
{}
### Response:
"""
        
        formatted_prompt = alpaca_template.format(prompt)
        
        # Tokenize input
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        # Generate response
        print(f"📝 Generating response for: {prompt}")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part (after "### Response:")
        if "### Response:" in full_response:
            response_text = full_response.split("### Response:")[1].strip()
        else:
            response_text = full_response.strip()
            
        return response_text
    
    def chat_mode(self):
        """Interactive chat mode"""
        print("💬 Chat mode started! Type 'quit' to exit.")
        print("=" * 50)
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            try:
                response = self.generate_response(user_input)
                print(f"\n🤖 Twin: {response}")
            except Exception as e:
                print(f"❌ Error generating response: {e}")

def main():
    parser = argparse.ArgumentParser(description="Use DPO fine-tuned LLM twin model")
    parser.add_argument("--model-path", default="./dpo_llm_twin_merged", 
                       help="Path to the fine-tuned model")
    parser.add_argument("--prompt", type=str, help="Single prompt to process")
    parser.add_argument("--max-tokens", type=int, default=256, 
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="Generation temperature")
    parser.add_argument("--no-sample", action="store_true", 
                       help="Disable sampling (use greedy decoding)")
    parser.add_argument("--chat", action="store_true", 
                       help="Start interactive chat mode")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = DPOModelInference(args.model_path)
    
    try:
        # Load model
        inference.load_model()
        
        if args.chat:
            # Interactive chat mode
            inference.chat_mode()
        elif args.prompt:
            # Single prompt mode
            response = inference.generate_response(
                args.prompt, 
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                do_sample=not args.no_sample
            )
            print(f"\n🤖 Response: {response}")
        else:
            # Default: test with sample prompts
            test_prompts = [
                "What are your main areas of expertise and experience?",
                "Can you describe your approach to software development?",
                "What technologies do you work with most frequently?"
            ]
            
            print("🧪 Testing model with sample prompts:")
            print("=" * 50)
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\n--- Test {i} ---")
                response = inference.generate_response(prompt)
                print(f"Prompt: {prompt}")
                print(f"Response: {response}")
                print()
                
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
