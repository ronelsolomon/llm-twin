#!/usr/bin/env python3
"""
Check MMLU-Pro dataset format to understand structure
"""

from datasets import load_dataset

def check_mmlu_format():
    """Check the actual format of MMLU-Pro dataset"""
    print("Loading MMLU-Pro dataset...")
    
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        print(f"Dataset loaded successfully with {len(dataset)} examples")
        
        # Check first few examples
        print("\nFirst example structure:")
        first_example = dataset[0]
        print(f"Keys: {list(first_example.keys())}")
        
        print("\nFirst example content:")
        for key, value in first_example.items():
            print(f"{key}: {value}")
            
        # Check if choices exist in any example
        print("\nChecking for 'choices' field in first 10 examples...")
        found_choices = False
        for i in range(min(10, len(dataset))):
            example = dataset[i]
            if 'choices' in example:
                print(f"Example {i} has choices: {example['choices']}")
                found_choices = True
                break
        
        if not found_choices:
            print("No 'choices' field found in first 10 examples")
            
            # Check what fields are available
            print("\nAvailable fields in first 5 examples:")
            for i in range(min(5, len(dataset))):
                example = dataset[i]
                print(f"Example {i}: {list(example.keys())}")
                
                # Look for any field that might contain choices
                for key, value in example.items():
                    if isinstance(value, list) and len(value) > 1:
                        print(f"  {key}: {value[:3]}...")  # Show first 3 items
                        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        
        # Try regular MMLU instead
        print("\nTrying regular MMLU dataset...")
        try:
            dataset = load_dataset("cais/mmlu", "all", split="test")
            print(f"Regular MMLU loaded with {len(dataset)} examples")
            
            first_example = dataset[0]
            print(f"Keys: {list(first_example.keys())}")
            print("\nFirst example:")
            for key, value in first_example.items():
                print(f"{key}: {value}")
        except Exception as e2:
            print(f"Error with regular MMLU: {e2}")

if __name__ == "__main__":
    check_mmlu_format()
