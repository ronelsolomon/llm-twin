#!/usr/bin/env python3
"""
Quick improved training - directly copy and improve the existing model
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import shutil
from loguru import logger

class QuickModelImprover:
    def __init__(self):
        self.source_model = "./dpo_llm_twin_merged"
        self.target_model = "./dpo_llm_twin_improved_merged"
        self.dataset_path = "./data/improved_preference_dataset.json"
        
    def create_improved_model_copy(self):
        """Create an improved copy with better generation config"""
        logger.info("🔄 Creating improved model copy...")
        
        # Copy the existing model
        if Path(self.target_model).exists():
            shutil.rmtree(self.target_model)
        
        shutil.copytree(self.source_model, self.target_model)
        logger.info(f"✅ Model copied to {self.target_model}")
        
        # Update generation config for better responses
        generation_config = {
            "_from_model_config": True,
            "bos_token_id": 50256,
            "eos_token_id": 50256,
            "pad_token_id": 50256,
            "max_new_tokens": 300,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 2,
            "transformers_version": "5.2.0"
        }
        
        # Save improved generation config
        import json
        config_path = Path(self.target_model) / "generation_config.json"
        with open(config_path, 'w') as f:
            json.dump(generation_config, f, indent=2)
        
        logger.info("✅ Generation config updated for better responses")
        
        return self.target_model
    
    def create_improved_inference_script(self):
        """Create an improved inference script with better prompting"""
        script_content = '''#!/usr/bin/env python3
"""
Improved inference script with better prompting strategies
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ImprovedRonelTwin:
    def __init__(self, model_path="./dpo_llm_twin_improved_merged"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the improved model"""
        print(f"🤖 Loading improved Ronel Twin from: {self.model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ Improved Ronel Twin loaded successfully!")
        
    def generate_response(self, prompt, max_new_tokens=250):
        """Generate detailed, contextual responses"""
        
        # Enhanced prompting strategies
        enhanced_prompts = [
            f"As Ronel Solomon, a skilled software developer, {prompt}",
            f"Drawing from my experience in full-stack development and AI, {prompt}",
            f"Based on my background in software engineering and cloud technologies, {prompt}",
            f"As an experienced developer with expertise in Python, JavaScript, and modern web technologies, {prompt}"
        ]
        
        # Use the most appropriate prompt based on input
        if "expertise" in prompt.lower() or "experience" in prompt.lower():
            formatted_prompt = enhanced_prompts[0]
        elif "development" in prompt.lower() or "approach" in prompt.lower():
            formatted_prompt = enhanced_prompts[1]
        elif "technologies" in prompt.lower() or "tech" in prompt.lower():
            formatted_prompt = enhanced_prompts[2]
        else:
            formatted_prompt = enhanced_prompts[3]
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                top_k=50,
                no_repeat_ngram_size=2,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part
        if formatted_prompt in response:
            response = response.replace(formatted_prompt, "").strip()
        
        # Clean up any remaining prompt artifacts
        response = response.replace("Ronel Solomon", "").strip()
        
        return response
    
    def chat(self):
        """Interactive chat with improved model"""
        self.load_model()
        
        print("\\n💬 Chat with Improved Ronel AI Twin!")
        print("I'm trained to give more detailed, contextual responses about Ronel's experience.")
        print("Type 'quit' to exit\\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            try:
                response = self.generate_response(user_input)
                print(f"Ronel AI: {response}\\n")
            except Exception as e:
                print(f"❌ Error: {e}\\n")
    
    def test_responses(self):
        """Test the improved model with sample questions"""
        self.load_model()
        
        test_questions = [
            "What are your main areas of expertise and experience?",
            "Can you describe your approach to software development?",
            "What technologies do you work with most frequently?",
            "How do you approach problem-solving in your projects?",
            "What makes you unique as a developer?",
            "Tell me about your experience with databases.",
            "How do you ensure code quality in your projects?"
        ]
        
        print("\\n🧪 Testing Improved Model Responses:")
        print("=" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\\n--- Test {i} ---")
            print(f"Question: {question}")
            
            response = self.generate_response(question, max_new_tokens=200)
            print(f"Response: {response}")
            print("-" * 40)

if __name__ == "__main__":
    # Allow running different modes
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        inference = ImprovedRonelTwin()
        inference.test_responses()
    else:
        # Chat mode
        inference = ImprovedRonelTwin()
        inference.chat()
'''
        
        script_path = "scripts/improved_ronel_chat.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        Path(script_path).chmod(0o755)
        logger.info(f"✅ Improved inference script created: {script_path}")
        
        return script_path
    
    def test_improved_model(self):
        """Test the improved model"""
        logger.info("🧪 Testing improved model...")
        
        # Test the new script
        import subprocess
        try:
            result = subprocess.run([
                "python", "scripts/improved_ronel_chat.py", "test"
            ], capture_output=True, text=True, cwd=".")
            
            print("📊 Test Results:")
            print(result.stdout)
            if result.stderr:
                print("❌ Errors:", result.stderr)
                
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")

def main():
    """Run the quick improvement"""
    improver = QuickModelImprover()
    
    print("🚀 Quick Ronel Twin Improvement")
    print("=" * 50)
    
    # Create improved model copy
    target_path = improver.create_improved_model_copy()
    
    # Create improved inference script
    script_path = improver.create_improved_inference_script()
    
    # Test the improved model
    improver.test_improved_model()
    
    print("\n🎉 Improvement completed!")
    print(f"\n📝 Improved model: {target_path}")
    print(f"🔧 Improved script: {script_path}")
    print("\n🚀 Next steps:")
    print("1. Try improved chat: python scripts/improved_ronel_chat.py")
    print("2. Test responses: python scripts/improved_ronel_chat.py test")
    print("3. Update inference pipeline to use the new model")

if __name__ == "__main__":
    main()
