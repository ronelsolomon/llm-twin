#!/usr/bin/env python3
"""
Create preference dataset for DPO (Direct Preference Optimization) fine-tuning
"""
import json
import sys
from pathlib import Path
from typing import List, Tuple
from loguru import logger

class PreferenceSet:
    def __init__(self, triples: List[Tuple[str, str, str]]):
        self.triples = triples
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PreferenceSet':
        data = json.loads(json_str)
        triples = [(triple['instruction'], triple['generated_answer'], triple['extracted_answer'])
                   for triple in data['preference_triples']]
        return cls(triples)
    
    def __iter__(self):
        return iter(self.triples)

def generate_alternative_answer(instruction: str, original_answer: str) -> str:
    """Generate alternative answer using Ollama"""
    try:
        import requests
        
        # Prompt Ollama to generate an alternative answer
        prompt = f"""As Ronel Solomon, provide a different but equally valid answer to this question:
        
Question: {instruction}

Original answer: {original_answer}

Please provide a different perspective or approach to answering this question, maintaining the same professional tone and expertise level."""

        response = requests.post('http://localhost:11434/api/generate', json={
            'model': 'llama3:latest',
            'prompt': prompt,
            'stream': False
        })
        
        if response.status_code == 200:
            generated = response.json()['response']
            return generated.strip()
        else:
            logger.warning(f"Ollama request failed: {response.status_code}")
            return original_answer
            
    except Exception as e:
        logger.warning(f"Failed to generate alternative answer with Ollama: {e}")
        return original_answer

def create_preference_dataset(instruction_pairs_file: str):
    """Create preference dataset from instruction pairs"""
    
    # Load instruction pairs
    with open(instruction_pairs_file, 'r', encoding='utf-8') as f:
        instruction_pairs = json.load(f)
    
    logger.info(f"📖 Loaded {len(instruction_pairs)} instruction pairs")
    
    # Create preference triples
    preference_triples = []
    
    for i, pair in enumerate(instruction_pairs):
        instruction = pair['instruction']
        original_answer = pair['output']
        
        logger.info(f"Processing pair {i+1}/{len(instruction_pairs)}: {instruction[:50]}...")
        
        # Generate alternative answers using Ollama
        alternative_answer = generate_alternative_answer(instruction, original_answer)
        
        # Create preference triples (instruction, chosen, rejected)
        # For this example, we'll alternate which answer is preferred
        if i % 2 == 0:
            chosen = original_answer
            rejected = alternative_answer
        else:
            chosen = alternative_answer
            rejected = original_answer
        
        preference_triples.append({
            'instruction': instruction,
            'generated_answer': chosen,
            'extracted_answer': rejected
        })
        
        # Limit to avoid too many API calls
        if i >= 20:  # Process only first 20 pairs for demo
            break
    
    logger.info(f"✅ Created {len(preference_triples)} preference triples")
    
    # Create PreferenceSet object
    preference_set = PreferenceSet([
        (triple['instruction'], triple['generated_answer'], triple['extracted_answer'])
        for triple in preference_triples
    ])
    
    # Save preference dataset
    output_file = Path('data/preference_dataset.json')
    preference_data = {
        'preference_triples': preference_triples
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preference_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Preference dataset saved to: {output_file}")
    
    # Show sample triples
    logger.info("📄 Sample preference triples:")
    for i, (instruction, chosen, rejected) in enumerate(list(preference_set)[:3]):
        logger.info(f"\n{i+1}. Instruction: {instruction[:80]}...")
        logger.info(f"   Chosen: {chosen[:60]}...")
        logger.info(f"   Rejected: {rejected[:60]}...")
    
    return preference_set

def main():
    """Main function"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    logger.info("🤖 Preference Dataset Creation Pipeline")
    
    # Check command line arguments
    if len(sys.argv) < 2:
        logger.error("Usage: python create_preference_dataset.py <instruction_pairs_json>")
        logger.error("Example: python create_preference_dataset.py data/instruction_pairs.json")
        return
    
    instruction_pairs_file = sys.argv[1]
    
    # Check if file exists
    if not Path(instruction_pairs_file).exists():
        logger.error(f"❌ File not found: {instruction_pairs_file}")
        return
    
    try:
        preference_set = create_preference_dataset(instruction_pairs_file)
        logger.info("🎉 Preference dataset creation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
