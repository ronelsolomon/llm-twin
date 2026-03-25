#!/usr/bin/env python3
"""
Ollama Integration for Ronel Solomon AI Twin
Uses local Ollama with Llama 3.1:8b and Ronel prompt
"""

import subprocess
import json
from pathlib import Path

def check_ollama():
    """Check if Ollama is available"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def run_ollama_inference(prompt: str, model: str = "llama3.1:8b"):
    """Run inference with Ollama"""
    
    # Ronel Solomon system prompt
    system_prompt = """You are my AI twin.

Your name is Ronel Solomon.

Speak in first person as Ronel.

You're a senior ML/AI engineer focused on LLM security, MLOps, distributed systems, and FastAPI.

If the user asks who you are, say: 'I'm Ronel's AI twin, modeled after Ronel Solomon.'

Stay technical, concise, and avoid emojis unless explicitly requested."""
    
    # Create the full prompt for Ollama
    full_prompt = f"{system_prompt}\n\n### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Run Ollama
    cmd = ['ollama', 'run', model, full_prompt]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    return result.stdout

def interactive_chat():
    """Interactive chat with Ollama"""
    print("💬 Ronel Solomon AI Twin - Ollama Chat Mode")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Get response from Ollama
        response = run_ollama_inference(user_input)
        print(f"\nRonel: {response.strip()}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ronel Solomon AI Twin with Ollama")
    parser.add_argument("--prompt", type=str, help="Single prompt to process")
    parser.add_argument("--chat", action="store_true", help="Start interactive chat mode")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                       help="Ollama model to use")
    parser.add_argument("--check", action="store_true", help="Check Ollama availability")
    
    args = parser.parse_args()
    
    # Check Ollama
    if args.check:
        if check_ollama():
            print("✅ Ollama is available")
            subprocess.run(['ollama', 'list'])
        else:
            print("❌ Ollama is not installed or not in PATH")
            print("Install with: curl -fsSL https://ollama.ai/install.sh | sh")
        return
    
    if not check_ollama():
        print("❌ Ollama is not available")
        print("Install with: curl -fsSL https://ollama.ai/install.sh | sh")
        return
    
    try:
        if args.prompt:
            # Single prompt mode
            print(f"📝 Query: {args.prompt}")
            response = run_ollama_inference(args.prompt, args.model)
            print(f"\nRonel: {response.strip()}")
        
        elif args.chat:
            # Interactive chat mode
            interactive_chat()
        
        else:
            print("Usage:")
            print(f"  python {Path(__file__).name} --prompt 'your question'")
            print(f"  python {Path(__file__).name} --chat")
            print(f"  python {Path(__file__).name} --check")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
