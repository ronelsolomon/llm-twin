#!/usr/bin/env python3
"""
Inference script for testing the trained Unsloth model
This script demonstrates how to use the trained model for inference.
"""
import sys
from pathlib import Path
from loguru import logger
import torch
from transformers import TextStreamer

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Unsloth
try:
    from unsloth import FastLanguageModel
except ImportError:
    logger.error("❌ Unsloth not installed. Install with: pip install unsloth")
    sys.exit(1)


def load_trained_model(model_path: str):
    """
    Load a trained model for inference.
    
    Args:
        model_path: Path to the trained model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"🤖 Loading trained model from {model_path}")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(model_path)
        FastLanguageModel.for_inference(model)
        logger.info("✅ Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def test_inference(model, tokenizer, instruction: str, max_new_tokens: int = 256, use_cache: bool = True):
    """
    Test inference with the trained model.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        instruction: The instruction to test
        max_new_tokens: Maximum number of new tokens to generate
        use_cache: Whether to use KV cache for faster generation
    """
    # Alpaca prompt template
    alpaca_prompt = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:

{}
### Response:
{}"""
    
    logger.info(f"🧪 Testing inference with: {instruction}")
    
    # Format the message
    message = alpaca_prompt.format(instruction, "")
    
    # Tokenize and move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([message], return_tensors="pt").to(device)
    
    # Create text streamer for output
    text_streamer = TextStreamer(tokenizer)
    
    # Generate response
    logger.info("🤖 Generating response...")
    _ = model.generate(
        **inputs, 
        streamer=text_streamer, 
        max_new_tokens=max_new_tokens, 
        use_cache=use_cache
    )
    
    logger.info("✅ Inference completed")


def interactive_chat(model, tokenizer):
    """
    Interactive chat mode with the model.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
    """
    logger.info("💬 Starting interactive chat mode...")
    logger.info("Type 'quit' or 'exit' to end the conversation")
    
    while True:
        try:
            # Get user input
            instruction = input("\n👤 You: ").strip()
            
            if instruction.lower() in ['quit', 'exit', 'q']:
                logger.info("👋 Ending chat session")
                break
            
            if not instruction:
                continue
            
            # Generate response
            test_inference(model, tokenizer, instruction)
            
        except KeyboardInterrupt:
            logger.info("\n👋 Chat interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Error during chat: {e}")


def main():
    """Main function to run inference testing."""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    
    logger.info("🚀 Unsloth Model Inference Testing")
    
    # Check command line arguments
    if len(sys.argv) < 2:
        logger.error("Usage: python test_inference.py <model_path> [instruction]")
        logger.error("Examples:")
        logger.error("  python test_inference.py llama3_lora_model")
        logger.error("  python test_inference.py llama3_lora_model 'Explain machine learning'")
        logger.error("  python test_inference.py llama3_lora_model --interactive")
        return
    
    model_path = sys.argv[1]
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"❌ Model path not found: {model_path}")
        return
    
    try:
        # Load the trained model
        model, tokenizer = load_trained_model(model_path)
        
        # Check if interactive mode
        if len(sys.argv) > 2 and sys.argv[2] == "--interactive":
            interactive_chat(model, tokenizer)
        else:
            # Single instruction test
            instruction = "Write a paragraph to introduce supervised fine-tuning."
            if len(sys.argv) > 2 and sys.argv[2] != "--interactive":
                instruction = sys.argv[2]
            
            test_inference(model, tokenizer, instruction)
        
        logger.info("🎉 Inference testing completed!")
        
    except Exception as e:
        logger.error(f"❌ Inference testing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
