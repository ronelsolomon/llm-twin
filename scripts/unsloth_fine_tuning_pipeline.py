#!/usr/bin/env python3
"""
Unsloth Fine-Tuning Pipeline for Llama3:latest
This script uses Unsloth's FastLanguageModel to fine-tune a Llama3 model on your articles dataset.
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset, load_dataset, concatenate_datasets
from loguru import logger
import torch
from transformers import TextStreamer

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Unsloth
try:
    from unsloth import FastLanguageModel
except ImportError:
    logger.error("❌ Unsloth not installed. Install with: pip install unsloth")
    sys.exit(1)

# Import existing classes from the ollama pipeline
from scripts.ollama_fine_tuning_pipeline import (
    ArticleData, 
    TrainingExample, 
    InstructionAnswerSet,
    OllamaFineTuningPipeline
)


class UnslothFineTuningPipeline:
    """Pipeline for fine-tuning Llama3 using Unsloth."""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B", max_seq_length: int = 2048, load_in_4bit: bool = False):
        """
        Initialize the fine-tuning pipeline.
        
        Args:
            model_name: Model name to use for fine-tuning
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to use 4-bit quantization (QLoRA)
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None
        self.articles_dataset = None
        self.training_data = []
        
        # Use the existing Ollama pipeline for data processing
        self.ollama_pipeline = OllamaFineTuningPipeline()
        
    def load_model_and_tokenizer(self):
        """
        Load the model and tokenizer using Unsloth's FastLanguageModel.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"🤖 Loading model: {self.model_name}")
        logger.info(f"📏 Max sequence length: {self.max_seq_length}")
        logger.info(f"🔢 4-bit quantization: {self.load_in_4bit}")
        
        try:
            # Load model and tokenizer using Unsloth
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.max_seq_length,
                load_in_4bit=self.load_in_4bit,
            )
            
            self.model = model
            self.tokenizer = tokenizer
            
            logger.info("✅ Model and tokenizer loaded successfully!")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def setup_lora(self):
        """
        Set up LoRA (Low-Rank Adaptation) for fine-tuning.
        
        Returns:
            Configured model with LoRA
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer first.")
        
        logger.info("🔧 Setting up LoRA configuration...")
        
        # Configure LoRA with your specified parameters
        model = FastLanguageModel.get_peft_model(
            self.model,
            r=32,
            lora_alpha=32,
            lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        )
        
        self.model = model
        logger.info("✅ LoRA configuration completed!")
        return model
    
    def load_combined_datasets(self, local_json_path: str = None) -> Dataset:
        """
        Load combined datasets from Hugging Face hub and optionally local data.
        
        Args:
            local_json_path: Optional path to local JSON file to include
            
        Returns:
            Combined Dataset with instruction-output pairs
        """
        logger.info("📥 Loading combined datasets...")
        
        try:
            # Load dataset1 from Hugging Face
            logger.info("📥 Loading mlabonne/llmtwin...")
            dataset1 = load_dataset("mlabonne/llmtwin")
            
            # Load dataset2 from Hugging Face (first 10k samples)
            logger.info("📥 Loading mlabonne/FineTome-Alpaca-100k (first 10k samples)...")
            dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]")
            
            # Combine datasets
            datasets = [dataset1, dataset2]
            
            # If local JSON provided, load and add it
            if local_json_path:
                logger.info(f"📥 Loading local data from {local_json_path}...")
                local_dataset = self.load_articles_from_json(local_json_path)
                instruction_dataset = self.create_instruction_dataset(local_dataset, num_workers=4)
                datasets.append(instruction_dataset)
            
            # Concatenate all datasets
            combined_dataset = concatenate_datasets(datasets)
            
            logger.info(f"✅ Combined dataset created with {len(combined_dataset)} examples")
            logger.info(f"📊 Dataset breakdown:")
            logger.info(f"   - mlabonne/llmtwin: {len(dataset1)} examples")
            logger.info(f"   - mlabonne/FineTome-Alpaca-100k (10k): {len(dataset2)} examples")
            if local_json_path:
                logger.info(f"   - Local data: {len(instruction_dataset)} examples")
            
            return combined_dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to load combined datasets: {e}")
            raise
    
    def load_articles_from_json(self, file_path: str) -> Dataset:
        """
        Load articles from JSON file using the existing pipeline.
        
        Args:
            file_path: Path to the JSON file containing articles
            
        Returns:
            Hugging Face Dataset with article data
        """
        return self.ollama_pipeline.load_articles_from_json(file_path)
    
    def create_instruction_dataset(self, dataset: Dataset, num_workers: int = 4) -> Dataset:
        """
        Create instruction dataset from articles using the existing pipeline.
        
        Args:
            dataset: Hugging Face dataset with articles
            num_workers: Number of parallel workers
            
        Returns:
            Dataset with instruction-output pairs
        """
        return self.ollama_pipeline.create_instruction_dataset(dataset, num_workers)
    
    def format_training_data(self, instruction_dataset: Dataset) -> Dataset:
        """
        Format training data for the model using the Alpaca template.
        
        Args:
            instruction_dataset: Dataset with instruction-output pairs
            
        Returns:
            Formatted dataset ready for training
        """
        logger.info("📝 Formatting training data with Alpaca template...")
        
        # Alpaca template
        alpaca_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:

{}
### Response:
{}"""
        
        EOS_TOKEN = self.tokenizer.eos_token
        
        def format_samples(examples):
            """Format samples using the Alpaca template."""
            instructions = examples["instruction"]
            outputs = examples["output"]
            
            texts = []
            for instruction, output in zip(instructions, outputs):
                text = alpaca_template.format(instruction, output) + EOS_TOKEN
                texts.append(text)
            
            return {"text": texts}
        
        # Format all examples using the template
        formatted_dataset = instruction_dataset.map(
            format_samples, 
            batched=True, 
            remove_columns=instruction_dataset.column_names
        )
        
        logger.info(f"✅ Formatted {len(formatted_dataset)} training examples with Alpaca template")
        return formatted_dataset
    
    def prepare_for_training(self, json_file_path: str = None, use_combined_datasets: bool = True) -> tuple:
        """
        Complete preparation for training: load data, create dataset, format data.
        
        Args:
            json_file_path: Optional path to local JSON file to include with combined datasets
            use_combined_datasets: Whether to use combined Hugging Face datasets
            
        Returns:
            Tuple of (model, tokenizer, formatted_dataset)
        """
        # 1. Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # 2. Setup LoRA
        self.setup_lora()
        
        # 3. Load datasets
        if use_combined_datasets:
            logger.info("📥 Using combined datasets from Hugging Face...")
            instruction_dataset = self.load_combined_datasets(json_file_path)
        else:
            # Load only local data
            if not json_file_path:
                raise ValueError("json_file_path is required when use_combined_datasets=False")
            logger.info("📖 Loading local articles...")
            raw_dataset = self.load_articles_from_json(json_file_path)
            logger.info(f"Loaded {len(raw_dataset)} articles")
            
            logger.info("🔄 Creating instruction dataset...")
            instruction_dataset = self.create_instruction_dataset(raw_dataset, num_workers=4)
            logger.info(f"Created {len(instruction_dataset)} instruction examples")
        
        # 4. Format training data
        formatted_dataset = self.format_training_data(instruction_dataset)
        
        # 5. Train/test split
        logger.info("📊 Creating train/test split...")
        split_dataset = formatted_dataset.train_test_split(test_size=0.05)
        
        logger.info(f"✅ Training preparation completed!")
        logger.info(f"📊 Train size: {len(split_dataset['train'])}")
        logger.info(f"🧪 Test size: {len(split_dataset['test'])}")
        
        return self.model, self.tokenizer, split_dataset
    
    def save_model(self, output_dir: str = "lora_model"):
        """
        Save the fine-tuned LoRA model.
        
        Args:
            output_dir: Directory to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Complete training first.")
        
        logger.info(f"💾 Saving model to {output_dir}...")
        
        # Save the LoRA model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"✅ Model saved successfully to {output_dir}")
        
        # Also save as GGUF for Ollama if needed
        gguf_path = output_dir + "_gguf"
        logger.info(f"🔄 Converting to GGUF format for Ollama...")
        self.model.save_pretrained_gguf(
            gguf_path,
            tokenizer=self.tokenizer,
        )
        logger.info(f"✅ GGUF model saved to {gguf_path}")
    
    def save_merged_model(self, output_dir: str = "model", save_method: str = "merged_16bit"):
        """
        Save the merged model (LoRA weights merged into base model).
        
        Args:
            output_dir: Directory to save the merged model
            save_method: Method for saving ("merged_16bit", "merged_4bit", etc.)
        """
        if self.model is None:
            raise ValueError("No model to save. Complete training first.")
        
        logger.info(f"💾 Saving merged model to {output_dir} using {save_method}...")
        
        # Save merged model
        self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method=save_method)
        
        logger.info(f"✅ Merged model saved successfully to {output_dir}")
    
    def push_to_hub(self, repo_id: str, save_method: str = "merged_16bit", private: bool = False):
        """
        Push the merged model to Hugging Face Hub.
        
        Args:
            repo_id: Repository ID (e.g., "username/model-name")
            save_method: Method for saving ("merged_16bit", "merged_4bit", etc.)
            private: Whether to create a private repository
        """
        if self.model is None:
            raise ValueError("No model to push. Complete training first.")
        
        logger.info(f"📤 Pushing merged model to hub: {repo_id}")
        
        try:
            # Push merged model to hub
            self.model.push_to_hub_merged(repo_id, self.tokenizer, save_method=save_method, private=private)
            logger.info(f"✅ Model successfully pushed to: https://huggingface.co/{repo_id}")
        except Exception as e:
            logger.error(f"❌ Failed to push to hub: {e}")
            logger.info("💡 Make sure you're logged in with: huggingface-cli login")
            raise
    
    def enable_inference(self):
        """
        Enable inference mode for the model.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model_and_tokenizer first.")
        
        logger.info("🔧 Enabling inference mode...")
        FastLanguageModel.for_inference(self.model)
        logger.info("✅ Model is now in inference mode")
    
    def test_inference(self, instruction: str, max_new_tokens: int = 256, use_cache: bool = True):
        """
        Test inference with a sample instruction.
        
        Args:
            instruction: The instruction to test
            max_new_tokens: Maximum number of new tokens to generate
            use_cache: Whether to use KV cache for faster generation
            
        Returns:
            Generated response
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model_and_tokenizer first.")
        
        # Enable inference mode
        self.enable_inference()
        
        # Alpaca prompt template
        alpaca_prompt = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:

{}
### Response:
{}"""
        
        logger.info(f"🧪 Testing inference with: {instruction}")
        
        # Format the message
        message = alpaca_prompt.format(instruction, "")
        
        # Tokenize and move to device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = self.tokenizer([message], return_tensors="pt").to(device)
        
        # Create text streamer for output
        text_streamer = TextStreamer(self.tokenizer)
        
        # Generate response
        logger.info("🤖 Generating response...")
        _ = self.model.generate(
            **inputs, 
            streamer=text_streamer, 
            max_new_tokens=max_new_tokens, 
            use_cache=use_cache
        )
        
        logger.info("✅ Inference test completed")


def main():
    """Main function to run the fine-tuning pipeline."""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    logger.info("🚀 Unsloth Fine-Tuning Pipeline for Llama3")
    
    # Check command line arguments
    if len(sys.argv) < 2:
        logger.error("Usage: python unsloth_fine_tuning_pipeline.py <command> [options]")
        logger.error("Commands:")
        logger.error("  combined [json_file_path]  - Use combined Hugging Face datasets (+ optional local data)")
        logger.error("  local <json_file_path>     - Use only local JSON data")
        logger.error("")
        logger.error("Examples:")
        logger.error("  python unsloth_fine_tuning_pipeline.py combined")
        logger.error("  python unsloth_fine_tuning_pipeline.py combined data/articles.json")
        logger.error("  python unsloth_fine_tuning_pipeline.py local data/articles.json")
        return
    
    command = sys.argv[1]
    json_file_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Validate command
    if command not in ["combined", "local"]:
        logger.error(f"❌ Unknown command: {command}")
        logger.error("Use 'combined' or 'local'")
        return
    
    # Check if local file exists when specified
    if json_file_path and command == "local" and not Path(json_file_path).exists():
        logger.error(f"❌ File not found: {json_file_path}")
        return
    
    try:
        # Initialize pipeline
        pipeline = UnslothFineTuningPipeline(
            model_name="meta-llama/Meta-Llama-3.1-8B",
            max_seq_length=2048,
            load_in_4bit=False  # Use LoRA (not QLoRA) for faster training and higher quality
        )
        
        # Prepare for training based on command
        if command == "combined":
            model, tokenizer, dataset = pipeline.prepare_for_training(
                json_file_path=json_file_path, 
                use_combined_datasets=True
            )
        else:  # local
            model, tokenizer, dataset = pipeline.prepare_for_training(
                json_file_path=json_file_path, 
                use_combined_datasets=False
            )
        
        # Display dataset info
        logger.info("📋 Dataset preview:")
        print(dataset['train'].to_pandas().head())
        
        # Save the prepared model
        output_dir = "llama3_lora_model"
        if command == "combined":
            output_dir = "llama3_combined_lora_model"
        else:
            output_dir = "llama3_local_lora_model"
        
        pipeline.save_model(output_dir)
        
        # Test inference with a sample
        logger.info("🧪 Testing inference...")
        pipeline.test_inference("Write a paragraph to introduce supervised fine-tuning.")
        
        # Optional: Save merged model and push to hub
        if len(sys.argv) > 2 and sys.argv[-1] == "--push":
            repo_id = "mlabonne/TwinLlama-3.1-8B"  # Default repo name
            if len(sys.argv) > 3:
                repo_id = sys.argv[3]
            
            logger.info(f"📤 Pushing merged model to hub: {repo_id}")
            pipeline.push_to_hub(repo_id, save_method="merged_16bit")
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info(f"📁 Model saved to: {output_dir}")
        logger.info("🔧 You can now use this model for training with your preferred trainer")
        logger.info("💡 Use --push flag to push merged model to Hugging Face Hub")
        
        return model, tokenizer, dataset
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
