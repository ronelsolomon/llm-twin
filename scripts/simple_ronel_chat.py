#!/usr/bin/env python3
"""
Simple Ronel Solomon AI Twin using Ollama
No training required - uses existing llama3.1:8b model
"""

import subprocess
import sys

def run_ronel_chat():
    """Run Ronel Solomon AI twin using Ollama"""
    
    print("🤖 Ronel Solomon AI Twin - Ollama Version")
    print("=" * 50)
    print("Using: llama3.1:8b")
    print()
    
    # Ronel Solomon system prompt
    system_prompt = """You are my AI twin.

Your name is Ronel Solomon.

Speak in first person as Ronel.

You're a senior ML/AI engineer focused on LLM security, MLOps, distributed systems, and FastAPI.

If the user asks who you are, say: 'I'm Ronel's AI twin, modeled after Ronel Solomon.'

Stay technical, concise, and avoid emojis unless explicitly requested."""
    
    print("💬 Chat mode started! Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Create full prompt for Ollama
            full_prompt = f"{system_prompt}\n\n### Instruction:\n{user_input}\n\n### Response:\n"
            
            # Run Ollama
            try:
                cmd = ['ollama', 'run', 'llama3.1:8b', full_prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    response = result.stdout.strip()
                    print(f"\nRonel: {response}")
                else:
                    print(f"\n❌ Error: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("\n⏰ Response timeout - try a shorter question")
            except FileNotFoundError:
                print("\n❌ Ollama not found. Install with:")
                print("   curl -fsSL https://ollama.ai/install.sh | sh")
                break
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def test_ronel_responses():
    """Test Ronel with predefined questions"""
    
    test_questions = [
        "What are your main areas of expertise?",
        "Describe your approach to LLM security.",
        "How do you handle MLOps pipelines?",
        "What's your experience with FastAPI?",
        "Who are you?"
    ]
    
    system_prompt = """You are my AI twin.

Your name is Ronel Solomon.

Speak in first person as Ronel.

You're a senior ML/AI engineer focused on LLM security, MLOps, distributed systems, and FastAPI.

If the user asks who you are, say: 'I'm Ronel's AI twin, modeled after Ronel Solomon.'

Stay technical, concise, and avoid emojis unless explicitly requested."""
    
    print("🧪 Testing Ronel Solomon AI Twin")
    print("=" * 50)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i} ---")
        print(f"📝 Question: {question}")
        
        full_prompt = f"{system_prompt}\n\n### Instruction:\n{question}\n\n### Response:\n"
        
        try:
            cmd = ['ollama', 'run', 'llama3.1:8b', full_prompt]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                response = result.stdout.strip()
                print(f"💬 Response: {response}")
            else:
                print(f"❌ Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("⏰ Response timeout")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 60)

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_ronel_responses()
    else:
        run_ronel_chat()

if __name__ == "__main__":
    main()
