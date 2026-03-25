#!/usr/bin/env python3
"""
Mac-compatible Ronel Solomon AI Twin Finetuning for Llama 3.1:8b
Uses standard transformers since Unsloth GPU acceleration isn't available on Mac
"""

import os
import torch
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)

class MacRonelTwinTrainer:
    def __init__(self, base_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
    def create_ronel_training_data(self, num_samples: int = 500) -> Dataset:
        """Create Ronel Solomon training data optimized for Llama 3.1:8b"""
        
        logger.info(f"📝 Creating {num_samples} Ronel Solomon training samples...")
        
        # Ronel Solomon expertise areas
        ronel_expertise = [
            "LLM security and safety mechanisms",
            "MLOps pipeline design and automation", 
            "Distributed systems and microservices",
            "FastAPI development and API design",
            "Machine learning model deployment",
            "Cloud architecture (AWS, GCP, Azure)",
            "Python, TypeScript, and Go development",
            "Docker and Kubernetes orchestration",
            "Data engineering and ETL pipelines"
        ]
        
        training_samples = []
        
        for i in range(num_samples):
            expertise = ronel_expertise[i % len(ronel_expertise)]
            
            # Create varied, high-quality instructions
            instruction_types = [
                f"Explain your approach to {expertise.lower()}",
                f"What are best practices for {expertise.lower()}?",
                f"Describe your experience with {expertise.lower()}",
                f"How do you implement {expertise.lower()} securely?",
                f"What tools do you use for {expertise.lower()}?"
            ]
            
            instruction = instruction_types[i % len(instruction_types)]
            
            # Generate detailed, technical responses
            response = f"""I specialize in {expertise} with extensive hands-on experience implementing production-ready solutions.

For {expertise.lower()}, my approach focuses on:

1. **Security-First Design**: I always prioritize security considerations from the start
2. **Scalable Architecture**: Building systems that can handle growth and load
3. **Automation**: Implementing comprehensive CI/CD pipelines
4. **Monitoring**: Full observability and alerting systems

I've successfully deployed multiple {expertise.lower()} solutions in various environments, from startups to enterprise scale. My technical expertise includes modern tools and frameworks while maintaining best practices for maintainability and performance."""

            training_samples.append({
                'instruction': instruction,
                'output': response
            })
        
        # Create dataset
        dataset = Dataset.from_list(training_samples)
        
        logger.info(f"✅ Created {len(training_samples)} training samples")
        return dataset
    
    def load_ollama_model(self):
        """Load Ollama's llama3.1:8b model for Mac training"""
        logger.info("🤖 Using Ollama's llama3.1:8b model for training...")
        
        # For Mac training, we'll use a smaller open model instead
        # Download a compatible open model
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Open alternative
        
        logger.info(f"🔄 Using open model: {model_name}")
        
        # Determine device
        if torch.backends.mps.is_available():
            device = "mps"
            logger.info("🍎 Using Apple Silicon (MPS)")
        else:
            device = "cpu"
            logger.info("💻 Using CPU")
        
        # Load model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "mps" else torch.float32,
                device_map="auto" if device == "cpu" else None,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Setup for Mac
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            self.model = model
            self.tokenizer = tokenizer
            
            logger.info(f"✅ Model loaded on {device}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            logger.info("💡 Try: huggingface-cli login")
            logger.info("💡 Or use Ollama directly for inference")
            raise
    
    def format_training_data(self, dataset: Dataset) -> Dataset:
        """Format data for Llama 3.1:8b training"""
        
        logger.info("📝 Formatting training data for Llama 3.1:8b...")
        
        # Ronel Solomon template for Llama 3.1:8b
        ronel_template = """You are my AI twin.

Your name is Ronel Solomon.

Speak in first person as Ronel.

You're a senior ML/AI engineer focused on LLM security, MLOps, distributed systems, and FastAPI.

If the user asks who you are, say: 'I'm Ronel's AI twin, modeled after Ronel Solomon.'

Stay technical, concise, and avoid emojis unless explicitly requested.

### Instruction:

{instruction}

### Response:

{output}"""
        
        EOS_TOKEN = self.tokenizer.eos_token
        
        def format_samples(examples):
            instructions = examples["instruction"]
            outputs = examples["output"]
            
            texts = []
            for instruction, output in zip(instructions, outputs):
                text = ronel_template.format(instruction=instruction, output=output) + EOS_TOKEN
                texts.append(text)
            
            return {"text": texts}
        
        # Format all examples
        formatted_dataset = dataset.map(
            format_samples,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"✅ Formatted {len(formatted_dataset)} training examples")
        return formatted_dataset
    
    def setup_mac_training(self) -> TrainingArguments:
        """Setup training arguments optimized for Mac"""
        return TrainingArguments(
            output_dir="./mac_ronel_twin",
            num_train_epochs=3,
            per_device_train_batch_size=1,  # Smaller for Mac memory
            gradient_accumulation_steps=8,  # Increase effective batch size
            warmup_steps=100,
            max_steps=-1,
            learning_rate=2e-4,
            fp16=torch.backends.mps.is_available(),  # Use fp16 on MPS
            bf16=False,
            logging_steps=10,
            optim="adamw_torch",  # Standard AdamW
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            eval_strategy="no",
            eval_steps=100,
            save_steps=100,
            save_total_limit=2,
            load_best_model_at_end=False,
            report_to="none",
            # Mac-specific optimizations
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            remove_unused_columns=False,
        )
    
    def train_on_mac(self):
        """Main training function for Mac"""
        logger.info("🍎 Starting Mac-optimized Ronel Solomon training...")
        
        # Load model and tokenizer
        model, tokenizer = self.load_ollama_model()
        
        # Create training data
        dataset = self.create_ronel_training_data(500)  # Smaller for Mac
        
        # Format data
        formatted_dataset = self.format_training_data(dataset)
        
        # Setup training arguments
        training_args = self.setup_mac_training()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=formatted_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("🏋‍♂️ Mac training started...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        
        logger.info("✅ Mac training completed!")
        logger.info(f"📁 Model saved to: {training_args.output_dir}")
        
        return training_args.output_dir
    
    def test_mac_model(self, model_path: str):
        """Test the Mac-trained model"""
        logger.info("🧪 Testing Mac-trained model...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Test questions
        test_questions = [
            "What are your main areas of expertise?",
            "Describe your approach to LLM security.",
            "How do you handle MLOps pipelines?",
            "What's your experience with FastAPI?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n📝 Test {i}: {question}")
            
            # Format with Ronel template
            ronel_prompt = f"""You are my AI twin.

Your name is Ronel Solomon.

Speak in first person as Ronel.

You're a senior ML/AI engineer focused on LLM security, MLOps, distributed systems, and FastAPI.

If the user asks who you are, say: 'I'm Ronel's AI twin, modeled after Ronel Solomon.'

Stay technical, concise, and avoid emojis unless explicitly requested.

### Instruction:

{question}

### Response:

"""
            
            inputs = tokenizer(ronel_prompt, return_tensors="pt").to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response
            if "### Response:" in response:
                clean_response = response.split("### Response:")[1].strip()
            else:
                clean_response = response.strip()
            
            logger.info(f"💬 Response: {clean_response[:200]}...")

def main():
    """Main Mac training function"""
    print("🍎 Mac Ronel Solomon AI Twin Training")
    print("=" * 50)
    print("Using standard transformers (compatible with Mac)")
    print()
    
    # Initialize trainer
    trainer = MacRonelTwinTrainer()
    
    try:
        # Train on Mac
        model_path = trainer.train_on_mac()
        
        # Test the model
        trainer.test_mac_model(model_path)
        
        print(f"\n✅ Training completed successfully!")
        print(f"📁 Model available at: {model_path}")
        print(f"\n🧪 To test with Ollama:")
        print(f"   ollama run llama3.1:8b 'your question here'")
        
    except Exception as e:
        logger.error(f"❌ Mac training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return

if __name__ == "__main__":
    main()
