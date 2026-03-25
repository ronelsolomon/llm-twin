#!/usr/bin/env python3
"""
Quick test to verify evaluation fixes
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from run_evaluation import main

if __name__ == "__main__":
    # Quick test with smaller sample
    import argparse
    
    # Simulate command line arguments for quick test
    sys.argv = [
        "run_evaluation.py",
        "--eval-type", "arena",  # Only arena for quick test
        "--arena-models", 
            "/Users/ronel/Downloads/llm twin/dpo_llm_twin",  # DPO model
            "microsoft/DialoGPT-medium",  # Base model for comparison
        "--arena-names", "DPO Twin", "Base Model",
        "--arena-rounds", "1",  # Just 1 round for quick test
        "--output", "quick_test_results"
    ]
    
    main()
