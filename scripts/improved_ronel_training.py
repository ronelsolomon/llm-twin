#!/usr/bin/env python3
"""
Improved Ronel Solomon AI Twin Finetuning Script
Addresses issues with poor model training quality
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
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from unsloth import FastLanguageModel, is_bfloat16_supported

class ImprovedRonelTwinTrainer:
    def __init__(self, config_path: str = "configs/improved_training_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
    def load_config(self):
        """Load training configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"⚠️ Config file not found: {self.config_path}")
            return self.get_default_config()
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default training configuration"""
        return {
            'model': {
                'base_model': 'unsloth/Meta-Llama-3.1-8B-Instruct',
                'peft': {
                    'r': 64,
                    'lora_alpha': 16,
                    'lora_dropout': 0,
                    'bias': 'none',
                    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                    'use_gradient_checkpointing': 'unsloth',
                    'random_state': 3407,
                    'use_rslora': False,
                    'loft_q_config': None
                }
            },
            'training': {
                'output_dir': './improved_ronel_twin',
                'num_train_epochs': 3,
                'per_device_train_batch_size': 2,
                'gradient_accumulation_steps': 4,
                'warmup_steps': 100,
                'max_steps': -1,
                'learning_rate': 2e-4,
                'fp16': not is_bfloat16_supported(),
                'bf16': is_bfloat16_supported(),
                'logging_steps': 10,
                'optim': 'adamw_8bit',
                'weight_decay': 0.01,
                'lr_scheduler_type': 'linear',
                'seed': 3407,
                'output_dir': './improved_ronel_twin',
                'evaluation_strategy': 'steps',
                'eval_steps': 50,
                'save_steps': 50,
                'save_total_limit': 2,
                'load_best_model_at_end': True,
                'report_to': 'none'
            },
            'data': {
                'max_length': 2048,
                'num_samples': 1000,  # Increased for better training
                'use_existing_data': True,
                'data_file': 'data/instruction_pairs.json'
            }
        }
    
    def create_high_quality_training_data(self, num_samples: int = 1000) -> Dataset:
        """Create high-quality instruction-response pairs for Ronel Solomon AI twin"""
        
        logger.info(f"📝 Creating {num_samples} high-quality training samples...")
        
        # Load existing data if available
        existing_data = []
        data_file = Path(self.config['data']['data_file'])
        
        if data_file.exists() and self.config['data']['use_existing_data']:
            with open(data_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            logger.info(f"📊 Loaded {len(existing_data)} existing samples")
        
        # Generate additional high-quality samples
        ronel_expertise = [
            "LLM security and safety mechanisms",
            "MLOps pipeline design and automation", 
            "Distributed systems and microservices",
            "FastAPI development and API design",
            "Machine learning model deployment",
            "Cloud architecture (AWS, GCP, Azure)",
            "Python, TypeScript, and Go development",
            "Docker and Kubernetes orchestration",
            "Data engineering and ETL pipelines",
            "Model monitoring and observability"
        ]
        
        additional_samples = []
        
        # Generate diverse, high-quality samples
        for i in range(num_samples - len(existing_data)):
            expertise_area = ronel_expertise[i % len(ronel_expertise)]
            
            # Create varied instruction types
            instruction_types = [
                f"Explain your approach to {expertise_area.lower()}",
                f"What are the key challenges in {expertise_area.lower()}?",
                f"Describe a project where you implemented {expertise_area.lower()}",
                f"Best practices for {expertise_area.lower()}?",
                f"How do you stay current with {expertise_area.lower()} trends?"
            ]
            
            instruction = instruction_types[i % len(instruction_types)]
            
            # Generate detailed, technical responses
            response_templates = [
                f"""I specialize in {expertise_area} with extensive hands-on experience. 

For {expertise_area.lower()}, I focus on building scalable, production-ready systems. My approach emphasizes security first, then performance and maintainability. I've implemented numerous solutions in this domain, from small startups to enterprise-scale deployments.

Key aspects I prioritize:
- Security-by-design principles
- Automated testing and deployment
- Comprehensive monitoring and logging
- Cost optimization and resource efficiency
- Team collaboration and knowledge sharing""",
                
                f"""In my work with {expertise_area.lower()}, I've developed a systematic approach:

1. **Assessment**: Understanding requirements and constraints
2. **Architecture**: Designing scalable, secure solutions  
3. **Implementation**: Following best practices and coding standards
4. **Testing**: Comprehensive validation and edge case handling
5. **Deployment**: Automated, monitored rollouts
6. **Optimization**: Performance tuning and cost management

I've successfully delivered multiple projects in this area, focusing on reliability and business impact.""",
                
                f"""With {expertise_area}, I bring both technical depth and practical experience:

**Technical Skills:**
- Advanced system design patterns
- Security-first development methodology
- Performance optimization techniques
- Cloud-native architecture principles

**Project Experience:**
- Led enterprise-scale {expertise_area.lower()} initiatives
- Mentored teams on best practices
- Reduced infrastructure costs by 30-40%
- Improved system reliability and uptime

I stay current through continuous learning and hands-on experimentation."""
            ]
            
            response = response_templates[i % len(response_templates)]
            
            additional_samples.append({
                'instruction': instruction,
                'output': response
            })
        
        # Combine existing and new data
        all_samples = existing_data + additional_samples
        
        # Create dataset
        dataset = Dataset.from_list(all_samples)
        
        # Save for future use
        data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Created dataset with {len(all_samples)} samples")
        return dataset
    
    def format_training_data(self, dataset: Dataset) -> Dataset:
        """Format data for training with improved Ronel Solomon prompt"""
        
        logger.info("📝 Formatting training data with improved Ronel Solomon template...")
        
        # Improved Ronel Solomon template
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
            """Format samples using the improved template"""
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
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with Unsloth optimization"""
        model_config = self.config['model']
        
        logger.info(f"🤖 Loading model: {model_config['base_model']}")
        
        # Load with Unsloth for faster training
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_config['base_model'],
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=model_config['peft']['r'],
            lora_alpha=model_config['peft']['lora_alpha'],
            lora_dropout=model_config['peft']['lora_dropout'],
            target_modules=model_config['peft']['target_modules'],
            bias=model_config['peft']['bias'],
            use_gradient_checkpointing=model_config['peft']['use_gradient_checkpointing'],
            random_state=model_config['peft']['random_state'],
            use_rslora=model_config['peft']['use_rslora'],
            loft_q_config=model_config['peft']['loft_q_config'],
        )
        
        # Setup tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        
        self.model = model
        self.tokenizer = tokenizer
        
        logger.info("✅ Model and tokenizer loaded successfully!")
        return model, tokenizer
    
    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments with optimized parameters"""
        training_config = self.config['training']
        
        return TrainingArguments(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            warmup_steps=training_config['warmup_steps'],
            max_steps=training_config['max_steps'],
            learning_rate=training_config['learning_rate'],
            fp16=training_config['fp16'],
            bf16=training_config['bf16'],
            logging_steps=training_config['logging_steps'],
            optim=training_config['optim'],
            weight_decay=training_config['weight_decay'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            seed=training_config['seed'],
            evaluation_strategy=training_config['evaluation_strategy'],
            eval_steps=training_config['eval_steps'],
            save_steps=training_config['save_steps'],
            save_total_limit=training_config['save_total_limit'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            report_to=training_config['report_to'],
            # Enhanced parameters for better training
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            remove_unused_columns=False,
        )
    
    def train_model(self):
        """Main training function"""
        logger.info("🚀 Starting improved Ronel Solomon AI twin training...")
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Create or load training data
        num_samples = self.config['data']['num_samples']
        dataset = self.create_high_quality_training_data(num_samples)
        
        # Format data for training
        formatted_dataset = self.format_training_data(dataset)
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
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
            tokenizer=tokenizer,
        )
        
        # Start training
        logger.info("🏋‍♂️ Training started...")
        trainer.train()
        
        # Save final model
        trainer.save_model()
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"📁 Model saved to: {training_args.output_dir}")
        
        return training_args.output_dir
    
    def test_model(self, model_path: str):
        """Test the trained model with sample questions"""
        logger.info("🧪 Testing trained model...")
        
        # Load the trained model
        model, tokenizer = FastLanguageModel.from_pretrained(model_path)
        
        # Test questions
        test_questions = [
            "What are your main areas of expertise?",
            "Describe your approach to LLM security.",
            "How do you handle MLOps pipelines?",
            "What's your experience with FastAPI?",
            "Explain your distributed systems design philosophy."
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
            
            inputs = tokenizer(ronel_prompt, return_tensors="pt").to("cuda")
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response part
            if "### Response:" in response:
                clean_response = response.split("### Response:")[1].strip()
            else:
                clean_response = response.strip()
            
            logger.info(f"💬 Response: {clean_response[:200]}...")
            print("-" * 60)
    
    def save_config(self):
        """Save current configuration"""
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"💾 Configuration saved to {self.config_path}")

def main():
    """Main training function"""
    # Initialize trainer
    trainer = ImprovedRonelTwinTrainer()
    
    # Save default config
    trainer.save_config()
    
    logger.info("🎯 Starting Improved Ronel Solomon AI Twin Training")
    logger.info("=" * 60)
    
    try:
        # Train the model
        model_path = trainer.train_model()
        
        # Test the model
        trainer.test_model(model_path)
        
        logger.info("🎉 Training and testing completed successfully!")
        logger.info(f"📁 Model available at: {model_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
