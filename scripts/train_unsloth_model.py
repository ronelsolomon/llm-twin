#!/usr/bin/env python3
"""
Training script for the Unsloth Fine-Tuning Pipeline
This script shows how to actually train the model using SFTTrainer.
"""
import os
import sys
from pathlib import Path
from loguru import logger
from transformers import TrainingArguments
from datasets import Dataset
import torch

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Unsloth and SFTTrainer
try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
except ImportError:
    logger.error("❌ Unsloth or TRL not installed. Install with: pip install unsloth trl")
    sys.exit(1)

from scripts.unsloth_fine_tuning_pipeline import UnslothFineTuningPipeline


def is_bfloat16_supported():
    """Check if bfloat16 is supported on the current hardware."""
    return torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False


def train_model(pipeline: UnslothFineTuningPipeline, dataset: Dataset, output_dir: str = "output", max_seq_length: int = 2048):
    """
    Train the model using SFTTrainer with your specified configuration.
    
    Args:
        pipeline: The UnslothFineTuningPipeline instance
        dataset: The prepared dataset with train/test splits
        output_dir: Directory to save the trained model
        max_seq_length: Maximum sequence length for training
    """
    logger.info("🚀 Starting model training with SFTTrainer...")
    
    # Configure SFTTrainer with your exact parameters
    trainer = SFTTrainer(
        model=pipeline.model,
        tokenizer=pipeline.tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=3e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=10,
            output_dir=output_dir,
            report_to="comet_ml",
            seed=0,
            eval_strategy="epoch",  # Evaluate at the end of each epoch
            save_strategy="epoch",  # Save checkpoint at the end of each epoch
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        ),
    )
    
    # Start training
    logger.info("🏃‍♂️ Training started...")
    trainer.train()
    
    logger.info("✅ Training completed!")
    
    # Save the final model
    pipeline.save_model(output_dir)
    
    return trainer


def main():
    """Main function to run the complete training pipeline."""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    logger.info("🎯 Complete Unsloth Training Pipeline")
    
    # Check command line arguments
    if len(sys.argv) < 2:
        logger.error("Usage: python train_unsloth_model.py <command> [options] [output_dir]")
        logger.error("Commands:")
        logger.error("  combined [json_file_path]  - Use combined Hugging Face datasets (+ optional local data)")
        logger.error("  local <json_file_path>     - Use only local JSON data")
        logger.error("")
        logger.error("Examples:")
        logger.error("  python train_unsloth_model.py combined")
        logger.error("  python train_unsloth_model.py combined data/articles.json my_model")
        logger.error("  python train_unsloth_model.py local data/articles.json my_model")
        return
    
    command = sys.argv[1]
    json_file_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].endswith('_model') else None
    output_dir = sys.argv[-1] if len(sys.argv) > 2 and sys.argv[-1].endswith('_model') else "llama3_trained_model"
    
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
            load_in_4bit=False  # Use LoRA for faster training and higher quality
        )
        
        # Prepare for training based on command
        if command == "combined":
            logger.info("🔧 Preparing combined datasets and model...")
            model, tokenizer, dataset = pipeline.prepare_for_training(
                json_file_path=json_file_path, 
                use_combined_datasets=True
            )
        else:  # local
            logger.info("🔧 Preparing local data and model...")
            model, tokenizer, dataset = pipeline.prepare_for_training(
                json_file_path=json_file_path, 
                use_combined_datasets=False
            )
        
        # Train the model
        trainer = train_model(pipeline, dataset, output_dir, max_seq_length=2048)
        
        logger.info("🎉 Training pipeline completed successfully!")
        logger.info(f"📁 Trained model saved to: {output_dir}")
        logger.info("🧪 You can now test the model with your custom prompts")
        
        # Example of how to use the trained model
        logger.info("🔍 Example usage:")
        logger.info("from unsloth import FastLanguageModel")
        logger.info(f"model, tokenizer = FastLanguageModel.from_pretrained('{output_dir}')")
        logger.info("messages = [{'role': 'user', 'content': 'Your prompt here'}]")
        logger.info("inputs = tokenizer.apply_chat_template(messages, return_tensors='pt')")
        logger.info("outputs = model.generate(**inputs, max_new_tokens=64)")
        logger.info("response = tokenizer.decode(outputs[0], skip_special_tokens=True)")
        
        return trainer
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
