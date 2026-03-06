#!/usr/bin/env python3
"""
DPO Fine-Tuning Script for Mac (without GPU acceleration)
Uses standard transformers instead of Unsloth for Mac compatibility
"""
import os
import torch
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, 
    TextStreamer, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOConfig, DPOTrainer

class DPOTrainerPipelineMac:
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
                'base_model': 'meta-llama/Llama-2-7b-chat-hf',  # Use Hugging Face model for Mac
                'peft': {
                    'r': 32,
                    'lora_alpha': 32,
                    'lora_dropout': 0.1,
                    'target_modules': ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
                    'load_in_4bit': True  # Enable for Mac memory efficiency
                }
            },
            'training': {
                'max_seq_length': 1024,  # Smaller for Mac
                'per_device_train_batch_size': 1,  # Smaller batch for Mac
                'per_device_eval_batch_size': 1,
                'gradient_accumulation_steps': 16,  # Increase to compensate
                'learning_rate': 2e-6,
                'num_train_epochs': 1,
                'warmup_steps': 10,
                'weight_decay': 0.01,
                'optim': "adamw_torch",
                'lr_scheduler_type': "linear",
                'eval_strategy': "steps",
                'eval_steps': 0.2,
                'logging_steps': 1,
                'fp16': True,  # Use fp16 on Mac
                'bf16': False,
                'output_dir': "output",
                'report_to': "none",
                'seed': 0,
            },
            'dpo': {
                'beta': 0.5,
                'max_length': 512,  # Smaller for Mac
                'max_prompt_length': 512,
            },
            'data': {
                'preference_dataset_path': 'data/preference_dataset.json',
                'train_test_split': 0.05
            },
            'output': {
                'model_dir': './dpo_llm_twin_mac',
                'merged_model_dir': './dpo_llm_twin_mac_merged'
            }
        }
        
    def load_model_and_tokenizer(self):
        """Load the model for DPO training"""
        model_config = self.config['model']['peft']
        base_model = self.config['model']['base_model']
        max_seq_length = self.config['training']['max_seq_length']
        
        logger.info(f"🤖 Loading model: {base_model}")
        
        # Configure quantization for Mac
        if model_config['load_in_4bit']:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            bnb_config = None
        
        # Load model with CPU offloading
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "6GB", "cpu": "30GB"}  # Adjust based on your Mac's specs
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Configure PEFT
        peft_config = LoraConfig(
            r=model_config['r'],
            lora_alpha=model_config['lora_alpha'],
            lora_dropout=model_config['lora_dropout'],
            target_modules=model_config['target_modules'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Prepare model for PEFT
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, peft_config)
        
        logger.info("✅ Model and tokenizer loaded successfully")
        
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
        
        return split_dataset
        
    def setup_dpo_training(self):
        """Setup DPO training configuration"""
        logger.info("⚙️ Setting up DPO training configuration")
        
        # DPO configuration
        dpo_config = self.config['dpo']
        training_config = self.config['training']
        
        # Use DPOConfig directly
        self.dpo_args = DPOConfig(
            learning_rate=training_config['learning_rate'],
            lr_scheduler_type=training_config['lr_scheduler_type'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            num_train_epochs=training_config['num_train_epochs'],
            fp16=training_config['fp16'],
            bf16=training_config['bf16'],
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
        """Train model using DPO"""
        logger.info("🚀 Starting DPO training")
        
        # Load dataset with train/test split
        dataset = self.load_preference_dataset()
        
        # Initialize DPO trainer
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=None,  # Use same model as reference for PEFT
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
        
        # Save PEFT model
        dpo_trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save merged model
        logger.info("🔄 Saving merged model...")
        merged_output_dir = self.config['output']['merged_model_dir']
        
        # Merge and save
        merged_model = dpo_trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_output_dir)
        self.tokenizer.save_pretrained(merged_output_dir)
        
        logger.info(f"✅ DPO training completed!")
        logger.info(f"📁 PEFT model saved to: {output_dir}")
        logger.info(f"📁 Merged model saved to: {merged_output_dir}")
        
        return merged_output_dir
        
    def test_model(self, model_path: str = None, test_prompt: str = None):
        """Test DPO fine-tuned model"""
        logger.info("🧪 Testing DPO fine-tuned model")
        
        # Use default merged model path from config if not provided
        if model_path is None:
            model_path = self.config['output']['merged_model_dir']
        
        max_seq_length = self.config['training']['max_seq_length']
        
        # Check if model exists
        if not Path(model_path).exists():
            logger.error(f"❌ Model not found: {model_path}")
            logger.info("💡 Run DPO training first: python scripts/dpo_fine_tuning_mac.py")
            return None
        
        # Load fine-tuned model for inference
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
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
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            use_cache=True,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
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
            
            # Step 2: Setup DPO training
            self.setup_dpo_training()
            
            # Step 3: Train DPO model
            model_path = self.train_dpo_model()
            
            # Step 4: Test the model (optional)
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
    
    logger.info("🤖 DPO Fine-Tuning Pipeline for Mac")
    logger.info("=" * 50)
    
    # Initialize pipeline
    pipeline = DPOTrainerPipelineMac()
    
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
