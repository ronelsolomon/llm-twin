#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Fine-Tuning Script for LLM Twin
Based on the preference dataset created from your LLM twin data
"""
import os
import torch
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from datasets import Dataset, load_dataset
from transformers import TrainingArguments, TextStreamer
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchDPOTrainer
from trl import DPOConfig, DPOTrainer

# Apply DPO patch for notebook environments
PatchDPOTrainer()

class DPOTrainerPipeline:
    def __init__(self, config_path: str = "configs/dpo_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.model = None
        self.tokenizer = None
        self.dataset = None
        
    def load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"⚠️ Config file not found: {self.config_path}")
            logger.info("Using default configuration")
            return self.get_default_config()
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration"""
        return {
            'model': {
                'base_model': 'llama3:latest',
                'peft': {
                    'r': 64,
                    'lora_alpha': 64,
                    'lora_dropout': 0,
                    'target_modules': ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
                    'load_in_4bit': False
                }
            },
            'training': {
                'max_seq_length': 2048,
                'per_device_train_batch_size': 2,
                'gradient_accumulation_steps': 4,
                'learning_rate': 5e-5,
                'num_train_epochs': 3,
                'warmup_steps': 100,
                'save_steps': 100,
                'eval_steps': 100
            },
            'dpo': {
                'beta': 0.1,
                'max_prompt_length': 1024,
                'max_target_length': 1024
            },
            'data': {
                'preference_dataset_path': 'data/preference_dataset.json',
                'train_test_split': 0.05
            },
            'output': {
                'model_dir': './dpo_llm_twin',
                'checkpoints_dir': './dpo_results'
            }
        }
        
    def load_model_and_tokenizer(self):
        """Load the fine-tuned LLM twin model for DPO training"""
        model_path = self.config['model']['base_model']
        max_seq_length = self.config['training']['max_seq_length']
        load_in_4bit = self.config['model']['peft']['load_in_4bit']
        
        logger.info(f"🤖 Loading model: {model_path}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )
        
        logger.info("✅ Model and tokenizer loaded successfully")
        
    def prepare_model_for_peft(self):
        """Prepare model for PEFT with LoRA configuration"""
        logger.info("🔧 Preparing model for PEFT with LoRA")
        
        peft_config = self.config['model']['peft']
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=peft_config['r'],
            lora_alpha=peft_config['lora_alpha'],
            lora_dropout=peft_config['lora_dropout'],
            target_modules=peft_config['target_modules'],
        )
        
        logger.info("✅ Model prepared for PEFT")
        
    def load_preference_dataset(self):
        """Load and prepare the preference dataset"""
        preference_dataset_path = self.config['data']['preference_dataset_path']
        train_test_split = self.config['data']['train_test_split']
        
        logger.info(f"📖 Loading preference dataset from {preference_dataset_path}")
        
        # Check if file exists
        if not Path(preference_dataset_path).exists():
            logger.error(f"❌ Preference dataset not found: {preference_dataset_path}")
            logger.info("💡 Run 'python scripts/create_preference_dataset.py data/instruction_pairs.json' first")
            raise FileNotFoundError(f"Preference dataset not found: {preference_dataset_path}")
        
        # Load preference dataset
        with open(preference_dataset_path, 'r', encoding='utf-8') as f:
            preference_data = json.load(f)
        
        # Extract preference triples
        preference_triples = preference_data.get('preference_triples', [])
        logger.info(f"📊 Loaded {len(preference_triples)} preference triples")
        
        # Convert to DPO format
        dpo_data = []
        alpaca_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:
{}
### Response:
"""
        
        EOS_TOKEN = self.tokenizer.eos_token
        
        for triple in preference_triples:
            instruction = triple['instruction']
            chosen = triple['generated_answer']
            rejected = triple['extracted_answer']
            
            # Format according to DPO requirements
            formatted_prompt = alpaca_template.format(instruction)
            
            dpo_data.append({
                "prompt": formatted_prompt,
                "chosen": chosen + EOS_TOKEN,
                "rejected": rejected + EOS_TOKEN
            })
        
        # Create dataset
        dataset = Dataset.from_list(dpo_data)
        
        # Create train/test split
        split_dataset = dataset.train_test_split(test_size=train_test_split)
        
        logger.info(f"✅ Dataset prepared: {len(split_dataset['train'])} training samples, {len(split_dataset['test'])} test samples")
        logger.info(f"📝 Sample formatted data:")
        logger.info(f"   Prompt: {dpo_data[0]['prompt'][:100]}...")
        logger.info(f"   Chosen: {dpo_data[0]['chosen'][:80]}...")
        logger.info(f"   Rejected: {dpo_data[0]['rejected'][:80]}...")
        
        return split_dataset
        
    def setup_dpo_training(self):
        """Setup DPO training configuration"""
        logger.info("⚙️ Setting up DPO training configuration")
        
        # DPO configuration
        dpo_config = self.config['dpo']
        training_config = self.config['training']
        
        # Use DPOConfig directly like in your example
        self.dpo_args = DPOConfig(
            learning_rate=training_config['learning_rate'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            num_train_epochs=training_config['num_train_epochs'],
            fp16=not is_bfloat16_supported(),  # Set dynamically based on hardware
            bf16=is_bfloat16_supported(),     # Set dynamically based on hardware
            optim=training_config['optim'],
            weight_decay=training_config['weight_decay'],
            warmup_steps=training_config['warmup_steps'],
            output_dir=training_config['output_dir'],
            eval_strategy=training_config['eval_strategy'],
            eval_steps=training_config['eval_steps'],
            logging_steps=training_config['logging_steps'],
            report_to=training_config['report_to'],
            seed=training_config['seed'],
        )
        
        # Store DPO parameters
        self.beta = dpo_config['beta']
        self.max_length = dpo_config['max_length']
        self.max_prompt_length = dpo_config['max_prompt_length']
        
        logger.info("✅ DPO training configuration ready")
        
    def train_dpo_model(self):
        """Train the model using DPO"""
        logger.info("🚀 Starting DPO training")
        
        # Load dataset with train/test split
        dataset = self.load_preference_dataset()
        
        # Initialize DPO trainer exactly like your example
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Use the same model as reference for PEFT
            tokenizer=self.tokenizer,
            beta=self.beta,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            max_length=self.max_length,
            max_prompt_length=self.max_prompt_length,
            args=self.dpo_args,
        )
        
        # Train the model
        logger.info("📚 Training DPO model...")
        dpo_trainer.train()
        
        # Save the model
        logger.info("💾 Saving DPO fine-tuned model...")
        output_dir = self.config['output']['model_dir']
        merged_output_dir = self.config['output']['merged_model_dir']
        
        # Save using DPO trainer (LoRA weights)
        dpo_trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save merged model (16-bit) like in your example
        logger.info("🔄 Saving merged model (16-bit)...")
        self.model.save_pretrained_merged(merged_output_dir, self.tokenizer, save_method="merged_16bit")
        
        logger.info(f"✅ DPO training completed!")
        logger.info(f"📁 LoRA model saved to: {output_dir}")
        logger.info(f"📁 Merged model saved to: {merged_output_dir}")
        
        # Quick sanity check like in your example
        logger.info("🧪 Running quick sanity check...")
        self._quick_sanity_check()
        
        return merged_output_dir
    
    def _quick_sanity_check(self):
        """Quick sanity check after training (like in your example)"""
        # Prepare model for inference
        FastLanguageModel.for_inference(self.model)
        
        # Test prompt
        alpaca_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:
{}
### Response:
"""
        
        message = alpaca_template.format("Write a paragraph to introduce supervised fine-tuning.", "")
        inputs = self.tokenizer([message], return_tensors="pt").to("cuda")
        
        # Generate response
        text_streamer = TextStreamer(self.tokenizer)
        logger.info("🤖 Model response:")
        _ = self.model.generate(**inputs, streamer=text_streamer, max_new_tokens=256, use_cache=True)
        
    def test_model(self, model_path: str = None, test_prompt: str = None):
        """Test the DPO fine-tuned model"""
        logger.info("🧪 Testing DPO fine-tuned model")
        
        # Use default merged model path from config if not provided
        if model_path is None:
            model_path = self.config['output']['merged_model_dir']
        
        max_seq_length = self.config['training']['max_seq_length']
        
        # Check if model exists
        if not Path(model_path).exists():
            logger.error(f"❌ Model not found: {model_path}")
            logger.info("💡 Run DPO training first: python scripts/dpo_fine_tuning.py")
            return None
        
        # Load the fine-tuned model for inference
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=False,
        )
        
        # Enable fast inference
        FastLanguageModel.for_inference(model)
        
        # Test prompt
        if test_prompt is None:
            test_prompt = "What are your main areas of expertise and experience?"
        
        # Format prompt
        alpaca_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:
{}
### Response:
"""
        
        formatted_prompt = alpaca_template.format(test_prompt)
        
        # Generate response
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        response_text = response.split("### Response:")[1].strip()
        
        logger.info(f"📝 Test Prompt: {test_prompt}")
        logger.info(f"💬 Model Response: {response_text}")
        
        return response_text
        
    def run_pipeline(self, test_after_training: bool = True):
        """Run the complete DPO fine-tuning pipeline"""
        try:
            # Step 1: Load model and tokenizer
            self.load_model_and_tokenizer()
            
            # Step 2: Prepare model for PEFT
            self.prepare_model_for_peft()
            
            # Step 3: Load and prepare preference dataset
            self.load_preference_dataset()
            
            # Step 4: Setup DPO training
            self.setup_dpo_training()
            
            # Step 5: Train DPO model
            model_path = self.train_dpo_model()
            
            # Step 6: Test the model (optional)
            if test_after_training:
                self.test_model(model_path)
            
            logger.info("🎉 DPO fine-tuning pipeline completed successfully!")
            return model_path
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function"""
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level="INFO")
    
    logger.info("🤖 DPO Fine-Tuning Pipeline for LLM Twin")
    logger.info("=" * 50)
    
    # Initialize pipeline
    pipeline = DPOTrainerPipeline()
    
    # Run the complete pipeline
    try:
        model_path = pipeline.run_pipeline(test_after_training=True)
        logger.info(f"🎯 DPO fine-tuned model saved at: {model_path}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
