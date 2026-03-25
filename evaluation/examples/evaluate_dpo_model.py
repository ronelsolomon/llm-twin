#!/usr/bin/env python3
"""
Example: Evaluate your DPO fine-tuned LLM twin
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from run_evaluation import main

if __name__ == "__main__":
    # Example configuration for evaluating DPO model
    import argparse
    
    # Simulate command line arguments
    sys.argv = [
        "run_evaluation.py",
        "--eval-type", "both",
        "--mmlu-model", "/Users/ronel/Downloads/llm twin/dpo_llm_twin",  # Your DPO model path
        "--arena-models", 
            "/Users/ronel/Downloads/llm twin/dpo_llm_twin",  # DPO model
            "microsoft/DialoGPT-medium",  # Base model for comparison
        "--arena-names", "DPO Twin", "Base Model",
        "--arena-rounds", "2",
        "--mmlu-samples", "500",  # Test on 500 MMLU questions
        "--output", "evaluation_results"
    ]
    
    main()
