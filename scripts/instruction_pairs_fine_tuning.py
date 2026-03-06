#!/usr/bin/env python3
"""
Direct fine-tuning script using instruction pairs JSON file
"""
import json
import os
import sys
from pathlib import Path
from loguru import logger

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Main function to use instruction pairs for fine-tuning"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    logger.info("🤖 Instruction Pairs Fine-Tuning Pipeline")
    
    # Check command line arguments
    if len(sys.argv) < 2:
        logger.error("Usage: python instruction_pairs_fine_tuning.py <instruction_pairs_json>")
        logger.error("Example: python instruction_pairs_fine_tuning.py data/instruction_pairs.json")
        return
    
    json_file_path = sys.argv[1]
    
    # Check if file exists
    if not Path(json_file_path).exists():
        logger.error(f"❌ File not found: {json_file_path}")
        return
    
    try:
        # Load instruction pairs
        logger.info(f"📖 Loading instruction pairs from {json_file_path}")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            instruction_pairs = json.load(f)
        
        logger.info(f"✅ Loaded {len(instruction_pairs)} instruction pairs")
        
        # Display sample pairs
        logger.info("📄 Sample instruction pairs:")
        for i, pair in enumerate(instruction_pairs[:3]):
            logger.info(f"\n{i+1}. Instruction: {pair['instruction'][:80]}...")
            logger.info(f"   Output: {pair['output'][:80]}...")
        
        # Save in different formats for different fine-tuning approaches
        
        # 1. Save as simple JSON (already done)
        logger.info(f"✅ Instruction pairs ready in: {json_file_path}")
        
        # 2. Save as CSV for some fine-tuning frameworks
        import pandas as pd
        df = pd.DataFrame(instruction_pairs)
        csv_file = json_file_path.replace('.json', '.csv')
        df.to_csv(csv_file, index=False)
        logger.info(f"💾 Also saved as CSV: {csv_file}")
        
        # 3. Save in Alpaca format for Llama fine-tuning
        alpaca_data = []
        for pair in instruction_pairs:
            alpaca_data.append({
                "instruction": pair['instruction'],
                "input": "",
                "output": pair['output']
            })
        
        alpaca_file = json_file_path.replace('.json', '_alpaca.json')
        with open(alpaca_file, 'w', encoding='utf-8') as f:
            json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Also saved in Alpaca format: {alpaca_file}")
        
        logger.info("🎉 Instruction pairs processed successfully!")
        logger.info("\n📋 Next steps for fine-tuning:")
        logger.info("1. Use the JSON file directly with custom fine-tuning scripts")
        logger.info("2. Use the CSV file with frameworks that prefer CSV format")
        logger.info("3. Use the Alpaca format for Llama-based fine-tuning")
        logger.info("4. For Ollama: ollama create my-model -f ./Modelfile")
        logger.info("5. For Unsloth: python scripts/train_unsloth_model.py")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
