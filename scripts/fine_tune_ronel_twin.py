#!/usr/bin/env python3
"""
Fine-tuning script for Ronel's LLM twin.
This script uses Unsloth for efficient fine-tuning of a language model
to create a model that talks like Ronel Solomon.
"""

import os
import json
import torch
from datasets import Dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RonelTwinFineTuner:
    def __init__(self, 
                 model_name: str = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
                 dataset_path: str = "/Users/ronel/Downloads/llm twin/data/ronel_twin_dataset.json",
                 output_dir: str = "/Users/ronel/Downloads/llm twin/ronel_twin_model"):
        
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Model configuration
        self.max_seq_length = 2048
        self.dtype = None  # Auto-detect
        self.load_in_4bit = True
        
        # Training hyperparameters
        self.learning_rate = 2e-4
        self.batch_size = 2
        self.gradient_accumulation_steps = 4
        self.warmup_steps = 5
        self.max_steps = 100  # Small dataset, so fewer steps
        self.logging_steps = 1
        self.save_steps = 25
        
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer with Unsloth optimizations."""
        logger.info(f"Loading model: {self.model_name}")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
        )
        
        # Add chat template
        tokenizer = get_chat_template(
            tokenizer,
            chat_template="llama-3.1",
        )
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # LoRA rank
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        logger.info("Model loaded successfully")
        return model, tokenizer
    
    def load_and_preprocess_dataset(self):
        """Load the Ronel twin dataset and preprocess it for training."""
        logger.info(f"Loading dataset from: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to the format expected by the model
        formatted_data = []
        for entry in data:
            # Create a conversational format
            formatted_entry = {
                "messages": [
                    {"role": "user", "content": entry["instruction"]},
                    {"role": "assistant", "content": entry["output"]}
                ]
            }
            formatted_data.append(formatted_entry)
        
        # Create Hugging Face Dataset
        dataset = Dataset.from_list(formatted_data)
        
        # Apply chat template formatting
        def formatting_prompts_func(examples):
            convos = examples["messages"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) 
                    for convo in convos]
            return {"text": texts}
        
        dataset = dataset.map(formatting_prompts_func, batched=True)
        
        logger.info(f"Dataset loaded with {len(dataset)} examples")
        return dataset
    
    def setup_training_arguments(self):
        """Setup training arguments for fine-tuning."""
        return TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=self.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",  # Disable wandb/weights&biases for privacy
            save_steps=self.save_steps,
            save_total_limit=2,
        )
    
    def fine_tune(self):
        """Execute the fine-tuning process."""
        logger.info("Starting fine-tuning process...")
        
        # Load model and tokenizer
        global tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Load and preprocess dataset
        dataset = self.load_and_preprocess_dataset()
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Create trainer
        from trl import SFTTrainer
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=2,
            packing=False,  # Can be set to True for speedup
            args=training_args,
        )
        
        # Start training
        logger.info("Beginning training...")
        trainer.train()
        
        # Save the model
        logger.info(f"Saving model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        # Save model info
        model_info = {
            "base_model": self.model_name,
            "dataset_size": len(dataset),
            "training_steps": self.max_steps,
            "learning_rate": self.learning_rate,
            "created_at": str(torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"),
        }
        
        with open(os.path.join(self.output_dir, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("Fine-tuning completed successfully!")
        return model, tokenizer
    
    def test_model(self, model, tokenizer, test_prompts: list = None):
        """Test the fine-tuned model with sample prompts."""
        if test_prompts is None:
            test_prompts = [
                "Hi Ronel! Can you introduce yourself?",
                "What's your experience with machine learning?",
                "Tell me about your work at Berkeley.",
                "What technologies do you work with?",
                "What projects are you proud of?"
            ]
        
        logger.info("Testing the fine-tuned model...")
        
        FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
        
        for i, prompt in enumerate(test_prompts):
            messages = [
                {"role": "user", "content": prompt}
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=200,
                use_cache=True,
                temperature=0.7,
                do_sample=True,
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n--- Test {i+1} ---")
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print("-" * 50)

def main():
    """Main function to run the fine-tuning process."""
    # Check for GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name()}")
    else:
        logger.warning("No GPU available. Training will be slower on CPU.")
    
    # Initialize fine-tuner
    tuner = RonelTwinFineTuner()
    
    # Run fine-tuning
    model, tokenizer = tuner.fine_tune()
    
    # Test the model
    tuner.test_model(model, tokenizer)
    
    logger.info("Ronel twin fine-tuning completed!")

if __name__ == "__main__":
    main()
