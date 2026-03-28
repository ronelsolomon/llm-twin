#!/usr/bin/env python3
"""
MLX-based fine-tuning script for Ronel's LLM twin.
This script uses MLX framework which is optimized for Apple Silicon Macs.
"""

import os
import json
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from datasets import Dataset
import logging
from typing import Dict, List, Any
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RonelTwinMLXFineTuner:
    def __init__(self, 
                 model_name: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
                 dataset_path: str = "/Users/ronel/Downloads/llm twin/data/ronel_twin_dataset.json",
                 output_dir: str = "/Users/ronel/Downloads/llm twin/ronel_twin_model_mlx"):
        
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # MLX configuration
        self.max_seq_length = 2048
        self.batch_size = 2
        self.learning_rate = 1e-5
        self.num_epochs = 3
        self.warmup_steps = 10
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer using MLX."""
        logger.info(f"Loading MLX model: {self.model_name}")
        
        # Load model and tokenizer
        model, tokenizer = load(
            self.model_name,
            trust_remote_code=True
        )
        
        logger.info("MLX model loaded successfully")
        return model, tokenizer
    
    def load_and_preprocess_dataset(self, tokenizer):
        """Load and preprocess dataset for MLX training."""
        logger.info(f"Loading dataset from: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to MLX format
        training_data = []
        
        for entry in data:
            instruction = entry["instruction"]
            output = entry["output"]
            
            # Format as chat
            formatted_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
            
            training_data.append(formatted_text)
        
        logger.info(f"Dataset loaded with {len(training_data)} examples")
        return training_data
    
    def prepare_training_data(self, texts, tokenizer):
        """Prepare training data for MLX."""
        # Tokenize all texts
        tokenized_data = []
        
        for text in texts:
            # Tokenize
            tokens = tokenizer.encode(text)
            
            # Convert to MLX array
            if len(tokens) > self.max_seq_length:
                tokens = tokens[:self.max_seq_length]
            
            tokenized_data.append(mx.array(tokens))
        
        return tokenized_data
    
    def fine_tune(self):
        """Execute MLX fine-tuning process."""
        logger.info("Starting MLX fine-tuning process...")
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Load dataset
        training_texts = self.load_and_preprocess_dataset(tokenizer)
        
        # Prepare training data
        training_data = self.prepare_training_data(training_texts, tokenizer)
        
        # Fine-tuning configuration
        logger.info("Starting fine-tuning with MLX...")
        
        # Custom training loop for MLX
        optimizer = mx.optim.AdamW(learning_rate=self.learning_rate)
        
        # Convert to training mode
        model.train()
        
        # Training metrics
        total_steps = 0
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            
            # Process in batches
            for i in range(0, len(training_data), self.batch_size):
                batch_data = training_data[i:i + self.batch_size]
                
                # Pad sequences to same length
                max_len = max(len(seq) for seq in batch_data)
                padded_batch = []
                
                for seq in batch_data:
                    padded_seq = np.pad(seq, (0, max_len - len(seq)), constant_values=0)
                    padded_batch.append(mx.array(padded_seq))
                
                # Stack batch
                batch = mx.stack(padded_batch)
                
                # Forward pass
                inputs = batch[:, :-1]  # All but last token
                targets = batch[:, 1:]    # All but first token
                
                # Get model output
                def loss_fn(model, inputs, targets):
                    logits = model(inputs)
                    return nn.losses.cross_entropy(logits, targets)
                
                # Compute loss and gradients
                loss, grads = mx.value_and_grad(loss_fn)(model, inputs, targets)
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update model
                optimizer.update(model, grads)
                mx.eval(model, optimizer)
                
                total_steps += 1
                
                # Log progress
                if total_steps % 10 == 0:
                    logger.info(f"Step {total_steps}, Loss: {loss.item():.4f}")
        
            # Epoch summary
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        # Save the fine-tuned model
        logger.info(f"Saving fine-tuned model to {self.output_dir}")
        
        # Save model weights
        model.save_weights(os.path.join(self.output_dir, "model.safetensors"))
        
        # Save tokenizer
        tokenizer.save_pretrained(self.output_dir)
        
        # Save training info
        model_info = {
            "base_model": self.model_name,
            "dataset_size": len(training_texts),
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "framework": "mlx",
            "final_loss": avg_loss
        }
        
        with open(os.path.join(self.output_dir, "model_info.json"), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info("MLX fine-tuning completed successfully!")
        return model, tokenizer
    
    def test_model(self, model, tokenizer, test_prompts: List[str] = None):
        """Test the fine-tuned MLX model."""
        if test_prompts is None:
            test_prompts = [
                "Hi Ronel! Can you introduce yourself?",
                "What's your experience with machine learning?",
                "Tell me about your work at Berkeley.",
                "What technologies do you work with?",
                "What projects are you proud of?"
            ]
        
        logger.info("Testing fine-tuned MLX model...")
        
        for i, prompt in enumerate(test_prompts):
            # Format input
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            # Generate response
            response = generate(
                model,
                tokenizer,
                prompt=formatted_prompt,
                temp=0.7,
                max_tokens=200,
                verbose=False
            )
            
            print(f"\n--- Test {i+1} ---")
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print("-" * 50)

def main():
    """Main function to run MLX fine-tuning."""
    logger.info("Starting Ronel twin MLX fine-tuning...")
    
    # Check if MLX is available
    try:
        import mlx
        import mlx_lm
        logger.info("MLX and mlx-lm are available")
    except ImportError:
        logger.error("MLX or mlx-lm not installed. Please install with:")
        logger.error("pip install mlx mlx-lm")
        return
    
    # Initialize fine-tuner
    tuner = RonelTwinMLXFineTuner()
    
    # Run fine-tuning
    model, tokenizer = tuner.fine_tune()
    
    # Test model
    tuner.test_model(model, tokenizer)
    
    logger.info("Ronel twin MLX fine-tuning completed!")

if __name__ == "__main__":
    main()
