#!/usr/bin/env python3
"""
Test script to benchmark speculative decoding performance
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from evaluation.model_comparison_evaluator import ModelComparisonEvaluator
import time

def test_speculative_decoding():
    """Test speculative decoding vs regular generation"""
    
    print("="*80)
    print("Speculative Decoding Performance Test")
    print("="*80)
    
    # Model configurations
    model_configs = [
        {
            "name": "DPO-Llama",
            "path": "/Users/ronel/Downloads/llm twin/dpo_llm_twin"
        }
    ]
    
    # Test with speculative decoding enabled
    print("\n1. Testing WITH speculative decoding:")
    evaluator_spec = ModelComparisonEvaluator(model_configs, use_speculative=True)
    
    test_instruction = "Explain the benefits of renewable energy in simple terms. Write about 200 words."
    
    # Time multiple generations
    spec_times = []
    for i in range(3):
        start_time = time.time()
        response = evaluator_spec.generate_response("DPO-Llama", test_instruction, use_speculative=True)
        end_time = time.time()
        spec_times.append(end_time - start_time)
        print(f"  Generation {i+1}: {end_time - start_time:.2f}s")
        print(f"  Response: {response[:100]}...")
    
    avg_spec_time = sum(spec_times) / len(spec_times)
    print(f"  Average speculative time: {avg_spec_time:.2f}s")
    
    # Test without speculative decoding
    print("\n2. Testing WITHOUT speculative decoding:")
    evaluator_regular = ModelComparisonEvaluator(model_configs, use_speculative=False)
    
    regular_times = []
    for i in range(3):
        start_time = time.time()
        response = evaluator_regular.generate_response("DPO-Llama", test_instruction, use_speculative=False)
        end_time = time.time()
        regular_times.append(end_time - start_time)
        print(f"  Generation {i+1}: {end_time - start_time:.2f}s")
        print(f"  Response: {response[:100]}...")
    
    avg_regular_time = sum(regular_times) / len(regular_times)
    print(f"  Average regular time: {avg_regular_time:.2f}s")
    
    # Calculate speedup
    speedup = avg_regular_time / avg_spec_time if avg_spec_time > 0 else 1.0
    print(f"\n3. Performance Summary:")
    print(f"  Regular generation: {avg_regular_time:.2f}s")
    print(f"  Speculative generation: {avg_spec_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    if speedup > 1.1:
        print(f"  ✅ Speculative decoding improves performance by {(speedup-1)*100:.1f}%")
    elif speedup > 0.9:
        print(f"  ⚖️  Speculative decoding has similar performance")
    else:
        print(f"  ❌ Speculative decoding reduces performance by {(1-speedup)*100:.1f}%")
    
    # Test quality comparison
    print(f"\n4. Quality Comparison:")
    
    # Generate longer responses for quality testing
    long_instruction = """
    Write a detailed technical explanation of how machine learning works. Include:
    - Key concepts and terminology
    - Different types of machine learning
    - Practical examples
    - Common challenges and solutions
    Write about 400 words.
    """
    
    print("  Generating with speculative decoding...")
    spec_response = evaluator_spec.generate_response("DPO-Llama", long_instruction, use_speculative=True)
    print(f"  Speculative response length: {len(spec_response.split())} words")
    print(f"  Speculative response: {spec_response[:200]}...")
    
    print("\n  Generating without speculative decoding...")
    regular_response = evaluator_regular.generate_response("DPO-Llama", long_instruction, use_speculative=False)
    print(f"  Regular response length: {len(regular_response.split())} words")
    print(f"  Regular response: {regular_response[:200]}...")
    
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80)

if __name__ == "__main__":
    test_speculative_decoding()
