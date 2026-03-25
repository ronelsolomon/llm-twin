#!/usr/bin/env python3
"""
Demo: Test the training monitor with simulated data
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training_monitor import TrainingMonitor
import numpy as np

def demo_training_monitor():
    """Demonstrate training monitoring capabilities"""
    print("="*50)
    print("Training Monitor Demo")
    print("="*50)
    
    # Initialize monitor
    monitor = TrainingMonitor("demo_logs")
    
    # Simulate a realistic training scenario
    print("Simulating training...")
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")
        
        for step in range(50):
            global_step = epoch * 50 + step
            
            # Simulate realistic loss curves
            base_loss = 2.5 * np.exp(-global_step * 0.01)  # Exponential decay
            noise = np.random.normal(0, 0.02)  # Add noise
            train_loss = max(0.1, base_loss + noise)
            
            # Validation loss (slightly higher, with overfitting later)
            val_loss = train_loss + 0.15
            if epoch >= 1:  # Start overfitting in later epochs
                val_loss += (epoch - 1) * 0.05
            
            # Simulate gradient norm
            grad_norm = 2.0 * np.exp(-global_step * 0.005) + np.random.normal(0, 0.1)
            grad_norm = max(0.01, grad_norm)
            
            # Log every 5 steps
            if step % 5 == 0:
                monitor.log_training_step(
                    step=global_step,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=0.001 * (0.9 ** epoch),  # Learning rate decay
                    model=None  # Would pass actual model in real training
                )
    
    # Generate outputs
    print("\nGenerating training curves...")
    monitor.plot_training_curves()
    
    print("Saving metrics...")
    monitor.save_metrics()
    
    # Health check
    print("\n" + "="*50)
    print("Training Health Analysis")
    print("="*50)
    
    health = monitor.check_training_health()
    
    if health["issues"]:
        print("⚠️  Issues Detected:")
        for issue in health["issues"]:
            print(f"   • {issue}")
        print("\n💡 Recommendations:")
        for rec in health["recommendations"]:
            print(f"   • {rec}")
    else:
        print("✅ Training looks healthy!")
    
    # Summary stats
    summary = monitor.get_summary_stats()
    print(f"\n📊 Final Metrics:")
    print(f"   Training Loss: {summary['training_loss']['final']:.4f}")
    print(f"   Validation Loss: {summary['validation_loss']['final']:.4f}")
    print(f"   Perplexity: {summary['perplexity']['final']:.2f}")
    print(f"   Gradient Norm: {summary['gradient_norm']['final']:.4f}")
    
    print(f"\n📁 Results saved to: {monitor.output_dir}")
    print("="*50)

if __name__ == "__main__":
    demo_training_monitor()
