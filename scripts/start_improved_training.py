#!/usr/bin/env python3
"""
Quick start script for improved Ronel Solomon AI twin training
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.improved_ronel_training import ImprovedRonelTwinTrainer

def main():
    """Quick start training"""
    print("🚀 Starting Improved Ronel Solomon AI Twin Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = ImprovedRonelTwinTrainer()
    
    # Configuration options
    print("\n⚙️ Training Options:")
    print("1. Use default configuration (recommended)")
    print("2. Train with custom parameters")
    
    choice = input("\nSelect option (1-2): ").strip()
    
    if choice == "1":
        print("\n🎯 Using default configuration...")
        print("   - Base model: Llama-3.1-8B-Instruct")
        print("   - Training samples: 1000")
        print("   - Epochs: 3")
        print("   - LoRA rank: 64")
        print("   - Learning rate: 2e-4")
        
        try:
            # Train with default config
            model_path = trainer.train_model()
            print(f"\n✅ Training completed!")
            print(f"📁 Model saved to: {model_path}")
            print(f"\n🧪 To test the model:")
            print(f"   python scripts/use_improved_model.py --model {model_path}")
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            return False
            
    elif choice == "2":
        print("\n⚙️ Custom configuration not implemented yet.")
        print("   Edit configs/improved_training_config.yaml to customize")
        return False
        
    else:
        print("\n❌ Invalid choice")
        return False
    
    return True

if __name__ == "__main__":
    main()
