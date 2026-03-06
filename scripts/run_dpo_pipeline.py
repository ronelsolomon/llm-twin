#!/usr/bin/env python3
"""
Quick start script for DPO fine-tuning pipeline
This script orchestrates the complete DPO training process
"""
import sys
import subprocess
from pathlib import Path
from loguru import logger

def check_requirements():
    """Check if required files exist"""
    logger.info("🔍 Checking requirements...")
    
    # Check instruction pairs file
    instruction_pairs_file = Path("data/instruction_pairs.json")
    if not instruction_pairs_file.exists():
        logger.error(f"❌ Instruction pairs file not found: {instruction_pairs_file}")
        logger.info("💡 Make sure you have instruction pairs data in data/instruction_pairs.json")
        return False
    
    # Check preference dataset
    preference_dataset_file = Path("data/preference_dataset.json")
    if not preference_dataset_file.exists():
        logger.warning("⚠️ Preference dataset not found. Will create it first.")
        return "create_preference"
    
    logger.info("✅ All required files found")
    return True

def create_preference_dataset():
    """Create preference dataset from instruction pairs"""
    logger.info("📊 Creating preference dataset...")
    
    try:
        result = subprocess.run([
            sys.executable, "scripts/create_preference_dataset.py", 
            "data/instruction_pairs.json"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            logger.info("✅ Preference dataset created successfully")
            return True
        else:
            logger.error(f"❌ Failed to create preference dataset: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error creating preference dataset: {e}")
        return False

def run_dpo_training():
    """Run DPO fine-tuning"""
    logger.info("🚀 Starting DPO fine-tuning...")
    
    try:
        result = subprocess.run([
            sys.executable, "scripts/dpo_fine_tuning.py"
        ], cwd=".")
        
        if result.returncode == 0:
            logger.info("✅ DPO training completed successfully")
            return True
        else:
            logger.error(f"❌ DPO training failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running DPO training: {e}")
        return False

def main():
    """Main pipeline orchestrator"""
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level="INFO")
    
    logger.info("🤖 DPO Fine-Tuning Quick Start Pipeline")
    logger.info("=" * 50)
    
    # Step 1: Check requirements
    status = check_requirements()
    
    if status is False:
        logger.error("❌ Pipeline cannot start. Missing required files.")
        return 1
    
    # Step 2: Create preference dataset if needed
    if status == "create_preference":
        if not create_preference_dataset():
            logger.error("❌ Failed to create preference dataset")
            return 1
    
    # Step 3: Run DPO training
    if not run_dpo_training():
        logger.error("❌ DPO training failed")
        return 1
    
    logger.info("🎉 Complete DPO pipeline finished successfully!")
    logger.info("📁 Your DPO fine-tuned model is saved in: ./dpo_llm_twin/")
    logger.info("📖 Check DPO_TRAINING_GUIDE.md for usage instructions")
    
    return 0

if __name__ == "__main__":
    exit(main())
