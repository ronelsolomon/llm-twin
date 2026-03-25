#!/usr/bin/env python3
"""
Simple improved training script for DPO model
Uses basic transformers approach without complex dependencies
"""
import json
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from loguru import logger

class SimpleImprovedDPOTrainer:
    def __init__(self):
        self.base_model_path = "microsoft/DialoGPT-medium"
        self.output_dir = "./dpo_llm_twin_improved_merged"
        self.dataset_path = "./data/improved_preference_dataset.json"
        
    def load_improved_dataset(self):
        """Load the improved preference dataset"""
        logger.info(f"📝 Loading improved dataset from {self.dataset_path}")
        
        with open(self.dataset_path, 'r') as f:
            data = json.load(f)
        
        preference_triples = data['preference_triples']
        
        # Create training examples from preference data
        training_texts = []
        for triple in preference_triples:
            instruction = triple['instruction']
            preferred_response = triple['extracted_answer']
            
            # Format as conversational exchange
            formatted_text = f"User: {instruction}\nRonel: {preferred_response}\n"
            training_texts.append(formatted_text)
        
        logger.info(f"✅ Loaded {len(training_texts)} training examples")
        return training_texts
    
    def create_simple_dataset(self, texts):
        """Create a simple dataset for training"""
        class SimpleDataset(Dataset):
            def __init__(self, texts):
                self.texts = texts
                
            def __len__(self):
                return len(self.texts)
                
            def __getitem__(self, idx):
                return {"text": self.texts[idx]}
        
        return SimpleDataset(texts)
    
    def fine_tune_model(self):
        """Fine-tune the model with improved data"""
        logger.info("🚀 Starting improved fine-tuning...")
        
        # Load base model and tokenizer
        logger.info("📦 Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load training data
        training_texts = self.load_improved_dataset()
        dataset = self.create_simple_dataset(training_texts)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=5e-6,
            fp16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            report_to=None,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Start training
        logger.info("🏃 Starting training...")
        trainer.train()
        
        # Save model
        logger.info(f"💾 Saving improved model to {self.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)
        
        logger.info("✅ Training completed!")
        
        return model, tokenizer
    
    def test_improved_model(self, model, tokenizer):
        """Test the improved model"""
        logger.info("🧪 Testing improved model...")
        
        test_prompts = [
            "What are your main areas of expertise and experience?",
            "Can you describe your approach to software development?",
            "What technologies do you work with most frequently?",
            "How do you approach problem-solving in your projects?"
        ]
        
        for prompt in test_prompts:
            # Use conversational format
            formatted_prompt = f"User: {prompt}\nRonel:"
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    no_repeat_ngram_size=2,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part
            if formatted_prompt in response:
                response = response.replace(formatted_prompt, "").strip()
            
            print(f"\n📝 Prompt: {prompt}")
            print(f"💬 Response: {response}")
            print("-" * 50)
    
    def create_improved_inference_script(self):
        """Create an improved inference script"""
        script_content = '''#!/usr/bin/env python3
"""
Improved inference script for the better DPO model
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ImprovedDPOInference:
    def __init__(self, model_path="./dpo_llm_twin_improved_merged"):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the improved model"""
        print(f"🤖 Loading improved model from: {self.model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("✅ Improved model loaded successfully!")
        
    def generate_response(self, prompt, max_new_tokens=250):
        """Generate response with improved prompting"""
        
        # Use conversational context
        formatted_prompt = f"User: {prompt}\\nRonel:"
        
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
        
        return response
    
    def chat(self):
        """Interactive chat with improved model"""
        self.load_model()
        
        print("\\n💬 Chat with Improved Ronel AI Twin!")
        print("Type 'quit' to exit\\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("👋 Goodbye!")
                break
                
            try:
                response = self.generate_response(user_input)
                print(f"Ronel AI: {response}\\n")
            except Exception as e:
                print(f"❌ Error: {e}\\n")

if __name__ == "__main__":
    inference = ImprovedDPOInference()
    inference.chat()
'''
        
        script_path = "scripts/improved_dpo_chat.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        Path(script_path).chmod(0o755)
        logger.info(f"✅ Improved inference script created: {script_path}")

def main():
    """Run the improved training pipeline"""
    trainer = SimpleImprovedDPOTrainer()
    
    print("🚀 Simple Improved DPO Training")
    print("=" * 50)
    
    # Fine-tune model
    model, tokenizer = trainer.fine_tune_model()
    
    # Test the model
    trainer.test_improved_model(model, tokenizer)
    
    # Create improved inference script
    trainer.create_improved_inference_script()
    
    print("\n🎉 Improved training completed!")
    print("\n📝 Next steps:")
    print("1. Try the improved chat: python scripts/improved_dpo_chat.py")
    print("2. Update inference pipeline to use the new model")

if __name__ == "__main__":
    main()
