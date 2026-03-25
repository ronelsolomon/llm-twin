#!/usr/bin/env python3
"""
CPU-compatible fine-tuning script for Ronel's LLM twin.
This script uses standard transformers and PEFT for fine-tuning on CPU/Mac.
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import logging
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RonelTwinCPUFineTuner:
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",  # Smaller model for CPU
                 dataset_path: str = "/Users/ronel/Downloads/llm twin/data/ronel_twin_dataset.json",
                 output_dir: str = "/Users/ronel/Downloads/llm twin/ronel_twin_model_cpu"):
        
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Model configuration for CPU
        self.max_seq_length = 512  # Shorter for CPU efficiency
        self.device = "cpu"
        
        # Training hyperparameters
        self.learning_rate = 5e-5  # Lower learning rate for CPU
        self.batch_size = 1  # Small batch for CPU
        self.gradient_accumulation_steps = 8
        self.warmup_steps = 10
        self.max_steps = 200  # More steps for smaller batch size
        self.logging_steps = 5
        self.save_steps = 50
        
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer for CPU fine-tuning."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        # Setup LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", "c_fc"],  # GPT-style layers
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        logger.info("Model loaded successfully")
        return model, tokenizer
    
    def load_and_preprocess_dataset(self, tokenizer):
        """Load and preprocess Ronel twin dataset for training."""
        logger.info(f"Loading dataset from: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to conversational format
        formatted_texts = []
        for entry in data:
            # Create conversational format
            instruction = entry["instruction"]
            output = entry["output"]
            
            # Format as dialogue
            formatted_text = f"User: {instruction}\n\nRonel: {output}\n\n"
            formatted_texts.append(formatted_text)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.max_seq_length,
                return_tensors="pt"
            )
        
        # Create dataset
        dataset_dict = {"text": formatted_texts}
        dataset = Dataset.from_dict(dataset_dict)
        
        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        logger.info(f"Dataset loaded with {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def setup_training_arguments(self):
        """Setup training arguments for CPU fine-tuning."""
        return TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            fp16=False,  # No fp16 on CPU
            logging_steps=self.logging_steps,
            optim="adamw_torch",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",
            save_steps=self.save_steps,
            save_total_limit=3,
            dataloader_num_workers=0,  # CPU optimization
            remove_unused_columns=False,
        )
    
    def fine_tune(self):
        """Execute the fine-tuning process."""
        logger.info("Starting CPU fine-tuning process...")
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Load and preprocess dataset
        dataset = self.load_and_preprocess_dataset(tokenizer)
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # Start training
        logger.info("Beginning training...")
        trainer.train()
        
        # Save model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        # Save model info
        model_info = {
            "base_model": self.model_name,
            "dataset_size": len(dataset),
            "training_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "device": "cpu",
            "model_type": "peft_lora"
        }
        
        with open(os.path.join(self.output_dir, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("CPU fine-tuning completed successfully!")
        return model, tokenizer
    
    def test_model(self, model, tokenizer, test_prompts: List[str] = None):
        """Test the fine-tuned model."""
        if test_prompts is None:
            test_prompts = [
                "Hi Ronel! Can you introduce yourself?",
                "What's your experience with machine learning?",
                "Tell me about your work at Berkeley.",
                "What technologies do you work with?",
                "What projects are you proud of?"
            ]
        
        logger.info("Testing fine-tuned model...")
        
        model.eval()
        
        for i, prompt in enumerate(test_prompts):
            # Format input
            input_text = f"User: {prompt}\n\nRonel:"
            
            # Tokenize input
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_length
            )
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n--- Test {i+1} ---")
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print("-" * 50)

def main():
    """Main function to run CPU fine-tuning process."""
    logger.info("Starting Ronel twin CPU fine-tuning...")
    
    # Initialize fine-tuner
    tuner = RonelTwinCPUFineTuner()
    
    # Run fine-tuning
    model, tokenizer = tuner.fine_tune()
    
    # Test model
    tuner.test_model(model, tokenizer)
    
    logger.info("Ronel twin CPU fine-tuning completed!")

if __name__ == "__main__":
    main()
