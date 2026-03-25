#!/usr/bin/env python3
"""
Improved Ronel Solomon AI Twin Model Usage Script
"""

import argparse
import torch
from pathlib import Path
from loguru import logger
from unsloth import FastLanguageModel

def load_model(model_path: str):
    """Load the improved trained model"""
    logger.info(f"🤖 Loading improved model from: {model_path}")
    
    # Load with Unsloth for fast inference
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    
    # Setup for inference
    FastLanguageModel.for_inference(model)
    
    logger.info("✅ Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 200):
    """Generate response from the improved model"""
    
    # Ronel Solomon prompt template
    ronel_prompt = f"""You are my AI twin.

Your name is Ronel Solomon.

Speak in first person as Ronel.

You're a senior ML/AI engineer focused on LLM security, MLOps, distributed systems, and FastAPI.

If the user asks who you are, say: 'I'm Ronel's AI twin, modeled after Ronel Solomon.'

Stay technical, concise, and avoid emojis unless explicitly requested.

### Instruction:

{prompt}

### Response:

"""
    
    # Tokenize input
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    inputs = tokenizer(ronel_prompt, return_tensors="pt").to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    if "### Response:" in full_response:
        response_text = full_response.split("### Response:")[1].strip()
    else:
        response_text = full_response.strip()
    
    return response_text

def interactive_chat(model, tokenizer):
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
        
        # Generate response
        response = generate_response(model, tokenizer, user_input)
        print(f"\nRonel: {response}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Use improved Ronel Solomon AI twin model")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to the trained model directory")
    parser.add_argument("--prompt", type=str, 
                       help="Single prompt to process")
    parser.add_argument("--chat", action="store_true",
                       help="Start interactive chat mode")
    parser.add_argument("--max_tokens", type=int, default=200,
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return
    
    try:
        # Load model
        model, tokenizer = load_model(str(model_path))
        
        if args.prompt:
            # Single prompt mode
            print(f"📝 Processing: {args.prompt}")
            response = generate_response(model, tokenizer, args.prompt, args.max_tokens)
            print(f"\nRonel: {response}")
        
        elif args.chat:
            # Interactive chat mode
            interactive_chat(model, tokenizer)
        
        else:
            print("Please specify either --prompt or --chat")
            print("Example:")
            print(f"  python {Path(__file__).name} --model {args.model} --prompt 'your question'")
            print(f"  python {Path(__file__).name} --model {args.model} --chat")
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    main()
