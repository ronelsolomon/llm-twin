#!/usr/bin/env python3
"""
Export Ollama model to Hugging Face format for DPO training
"""
import subprocess
import os
from pathlib import Path
from loguru import logger

def export_ollama_model(ollama_model_name: str = "llm-twin", output_dir: str = "./hf_model"):
    """Export Ollama model to Hugging Face format"""
    
    logger.info(f"🔄 Exporting Ollama model '{ollama_model_name}' to Hugging Face format...")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Use ollama export command (if available) or alternative method
        # Note: This might require manual conversion depending on your Ollama setup
        
        # Method 1: Try ollama export (if supported)
        try:
            result = subprocess.run([
                "ollama", "export", ollama_model_name, "--output", output_dir
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Model exported to {output_dir}")
                return output_dir
        except FileNotFoundError:
            logger.warning("ollama export command not available")
        
        # Method 2: Manual export (requires finding model files)
        logger.info("🔍 Looking for Ollama model files...")
        
        # Find Ollama model directory
        ollama_home = os.path.expanduser("~/.ollama")
        models_dir = Path(ollama_home) / "models"
        
        if models_dir.exists():
            logger.info(f"📁 Found Ollama models directory: {models_dir}")
            # You would need to manually copy and convert the model files
            logger.warning("⚠️ Manual conversion required - see documentation")
        else:
            logger.error(f"❌ Ollama models directory not found at {models_dir}")
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Error exporting model: {e}")
        return None

def main():
    """Main function"""
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level="INFO")
    
    logger.info("🤖 Ollama to Hugging Face Export Tool")
    
    # Export the model
    output_dir = export_ollama_model()
    
    if output_dir:
        logger.info(f"✅ Model exported successfully to: {output_dir}")
        logger.info("💡 You can now use this path in DPO training")
    else:
        logger.error("❌ Model export failed")
        logger.info("💡 Consider using Option 2: Use a compatible base model instead")

if __name__ == "__main__":
    main()
