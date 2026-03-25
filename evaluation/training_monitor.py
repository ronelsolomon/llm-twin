"""
Training Metrics Monitor for LLM Twin
Tracks loss, perplexity, gradients during training
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import json
from pathlib import Path
from datetime import datetime
import wandb  # Optional: for experiment tracking

class TrainingMonitor:
    def __init__(self, output_dir: str = "training_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.metrics = {
            "training_loss": [],
            "validation_loss": [],
            "perplexity": [],
            "gradient_norm": [],
            "learning_rate": [],
            "epochs": [],
            "steps": []
        }
        
        self.start_time = datetime.now()
        
    def log_training_step(self, step: int, epoch: int, train_loss: float, 
                         val_loss: Optional[float] = None, 
                         model: Optional[torch.nn.Module] = None,
                         learning_rate: Optional[float] = None):
        """Log metrics for a training step"""
        
        # Calculate perplexity from training loss
        perplexity = np.exp(train_loss)
        
        # Calculate gradient norm if model provided
        grad_norm = 0.0
        if model is not None:
            total_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm ** (1. / 2)
        
        # Store metrics
        self.metrics["steps"].append(step)
        self.metrics["epochs"].append(epoch)
        self.metrics["training_loss"].append(train_loss)
        self.metrics["perplexity"].append(perplexity)
        self.metrics["gradient_norm"].append(grad_norm)
        self.metrics["learning_rate"].append(learning_rate)
        
        if val_loss is not None:
            self.metrics["validation_loss"].append(val_loss)
        
        # Print progress
        print(f"Step {step} | Epoch {epoch} | "
              f"Train Loss: {train_loss:.4f} | "
              f"PPL: {perplexity:.2f} | "
              f"Grad Norm: {grad_norm:.4f}")
        
        if val_loss is not None:
            print(f"  Val Loss: {val_loss:.4f}")
        
        # Optional: Log to wandb
        try:
            wandb.log({
                "step": step,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "perplexity": perplexity,
                "gradient_norm": grad_norm,
                "learning_rate": learning_rate
            })
        except:
            pass  # wandb not available
    
    def plot_training_curves(self, save_plots: bool = True):
        """Generate training curves plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics - {self.start_time.strftime("%Y-%m-%d %H:%M")}')
        
        # Training Loss
        axes[0, 0].plot(self.metrics["steps"], self.metrics["training_loss"], label="Training Loss")
        if self.metrics["validation_loss"]:
            val_steps = self.metrics["steps"][:len(self.metrics["validation_loss"])]
            axes[0, 0].plot(val_steps, self.metrics["validation_loss"], label="Validation Loss")
        axes[0, 0].set_title("Loss Curves")
        axes[0, 0].set_xlabel("Steps")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Perplexity
        axes[0, 1].plot(self.metrics["steps"], self.metrics["perplexity"], color='orange')
        axes[0, 1].set_title("Perplexity")
        axes[0, 1].set_xlabel("Steps")
        axes[0, 1].set_ylabel("Perplexity")
        axes[0, 1].grid(True)
        
        # Gradient Norm
        axes[1, 0].plot(self.metrics["steps"], self.metrics["gradient_norm"], color='green')
        axes[1, 0].set_title("Gradient Norm")
        axes[1, 0].set_xlabel("Steps")
        axes[1, 0].set_ylabel("Gradient Norm")
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # Learning Rate
        if any(lr is not None for lr in self.metrics["learning_rate"]):
            axes[1, 1].plot(self.metrics["steps"], self.metrics["learning_rate"], color='red')
            axes[1, 1].set_title("Learning Rate")
            axes[1, 1].set_xlabel("Steps")
            axes[1, 1].set_ylabel("Learning Rate")
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, "Learning Rate\nNot Tracked", 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {self.output_dir / 'training_curves.png'}")
        
        return fig
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        metrics_data = {
            "start_time": self.start_time.isoformat(),
            "metrics": self.metrics,
            "summary": self.get_summary_stats()
        }
        
        with open(self.output_dir / "training_metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        print(f"Training metrics saved to {self.output_dir / 'training_metrics.json'}")
    
    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics"""
        summary = {}
        
        if self.metrics["training_loss"]:
            summary["training_loss"] = {
                "final": self.metrics["training_loss"][-1],
                "min": min(self.metrics["training_loss"]),
                "max": max(self.metrics["training_loss"]),
                "mean": np.mean(self.metrics["training_loss"])
            }
        
        if self.metrics["validation_loss"]:
            summary["validation_loss"] = {
                "final": self.metrics["validation_loss"][-1],
                "min": min(self.metrics["validation_loss"]),
                "max": max(self.metrics["validation_loss"]),
                "mean": np.mean(self.metrics["validation_loss"])
            }
        
        if self.metrics["perplexity"]:
            summary["perplexity"] = {
                "final": self.metrics["perplexity"][-1],
                "min": min(self.metrics["perplexity"]),
                "max": max(self.metrics["perplexity"]),
                "mean": np.mean(self.metrics["perplexity"])
            }
        
        if self.metrics["gradient_norm"]:
            summary["gradient_norm"] = {
                "final": self.metrics["gradient_norm"][-1],
                "min": min(self.metrics["gradient_norm"]),
                "max": max(self.metrics["gradient_norm"]),
                "mean": np.mean(self.metrics["gradient_norm"])
            }
        
        return summary
    
    def check_training_health(self) -> Dict:
        """Check for common training issues"""
        issues = []
        recommendations = []
        
        # Check for overfitting
        if (len(self.metrics["training_loss"]) > 10 and 
            len(self.metrics["validation_loss"]) > 10):
            
            recent_train = np.mean(self.metrics["training_loss"][-5:])
            recent_val = np.mean(self.metrics["validation_loss"][-5:])
            
            if recent_val > recent_train * 1.1:
                issues.append("Potential overfitting detected")
                recommendations.append("Consider early stopping or regularization")
        
        # Check gradient explosion
        if self.metrics["gradient_norm"]:
            max_grad_norm = max(self.metrics["gradient_norm"])
            if max_grad_norm > 10.0:
                issues.append("Large gradient norms detected")
                recommendations.append("Consider gradient clipping or lower learning rate")
        
        # Check for vanishing gradients
        if self.metrics["gradient_norm"]:
            recent_grads = self.metrics["gradient_norm"][-10:]
            if np.mean(recent_grads) < 0.01:
                issues.append("Very small gradient norms")
                recommendations.append("Consider higher learning rate or different architecture")
        
        # Check perplexity trends
        if len(self.metrics["perplexity"]) > 20:
            recent_ppl = np.mean(self.metrics["perplexity"][-10:])
            early_ppl = np.mean(self.metrics["perplexity"][:10])
            
            if recent_ppl > early_ppl * 1.05:
                issues.append("Perplexity increasing")
                recommendations.append("Training may be diverging")
        
        return {
            "issues": issues,
            "recommendations": recommendations,
            "health_score": len(recommendations)  # Lower is better
        }

# Example usage during training
def example_training_loop():
    """Example of how to use TrainingMonitor during training"""
    monitor = TrainingMonitor("demo_training_logs")
    
    # Simulate training
    for epoch in range(5):
        for step in range(100):
            # Simulate metrics
            train_loss = 3.0 - (epoch * 0.3) - (step * 0.001) + np.random.normal(0, 0.05)
            val_loss = train_loss + 0.2 + np.random.normal(0, 0.02)
            
            # Log every 10 steps
            if step % 10 == 0:
                monitor.log_training_step(
                    step=epoch * 100 + step,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=0.001
                )
    
    # Generate plots and save
    monitor.plot_training_curves()
    monitor.save_metrics()
    
    # Check training health
    health = monitor.check_training_health()
    print("\nTraining Health Check:")
    for issue in health["issues"]:
        print(f"⚠️  {issue}")
    for rec in health["recommendations"]:
        print(f"💡 {rec}")

if __name__ == "__main__":
    example_training_loop()
