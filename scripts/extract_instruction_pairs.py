#!/usr/bin/env python3
"""
Extract instruction pairs from the Hugging Face dataset and save as simple JSON
"""
import json
from pathlib import Path
from datasets import load_dataset

def extract_instruction_pairs():
    """Extract instruction pairs from the dataset and save as simple JSON"""
    
    try:
        # Load the dataset
        dataset = load_dataset('./instruction_dataset.json')
        
        # Extract instruction-output pairs
        instruction_pairs = []
        
        # Get training data
        if 'train' in dataset:
            train_data = dataset['train'].to_pandas()
            for _, row in train_data.iterrows():
                if 'instruction' in row and 'output' in row:
                    instruction_pairs.append({
                        'instruction': row['instruction'],
                        'output': row['output']
                    })
        
        # Get test data
        if 'test' in dataset:
            test_data = dataset['test'].to_pandas()
            for _, row in test_data.iterrows():
                if 'instruction' in row and 'output' in row:
                    instruction_pairs.append({
                        'instruction': row['instruction'],
                        'output': row['output']
                    })
        
        # Save as simple JSON
        output_file = Path('data/instruction_pairs.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(instruction_pairs, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Extracted {len(instruction_pairs)} instruction pairs to {output_file}")
        
        # Show some examples
        if instruction_pairs:
            print("\n📄 Sample instruction pairs:")
            for i, pair in enumerate(instruction_pairs[:3]):
                print(f"\n{i+1}. Instruction: {pair['instruction'][:80]}...")
                print(f"   Output: {pair['output'][:80]}...")
        
        return instruction_pairs
        
    except Exception as e:
        print(f"❌ Error extracting pairs: {e}")
        return []

if __name__ == "__main__":
    extract_instruction_pairs()
