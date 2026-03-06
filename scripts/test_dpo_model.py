#!/usr/bin/env python3
"""
Test script for DPO fine-tuned LLM twin model
"""
import sys
from pathlib import Path
from loguru import logger
from unsloth import FastLanguageModel

def test_dpo_model(model_path: str = "./dpo_llm_twin"):
    """Test the DPO fine-tuned model with sample prompts"""
    
    # Check if model exists
    if not Path(model_path).exists():
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("💡 Run DPO training first: python scripts/dpo_fine_tuning.py")
        return False
    
    logger.info(f"🤖 Loading DPO model from: {model_path}")
    
    try:
        # Load model for inference
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            load_in_4bit=False,
        )
        
        # Enable fast inference
        FastLanguageModel.for_inference(model)
        
        # Test prompts
        test_prompts = [
            "What are your main areas of expertise and experience?",
            "Can you describe your approach to software development?",
            "What technologies do you work with most frequently?",
            "How do you approach problem-solving in your projects?",
            "What makes you unique as a developer?"
        ]
        
        alpaca_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:
{}
### Response:
"""
        
        logger.info("🧪 Testing DPO model with sample prompts:")
        logger.info("=" * 60)
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"\n📝 Test {i}: {prompt}")
            
            # Format prompt
            formatted_prompt = alpaca_template.format(prompt)
            
            # Generate response
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                use_cache=True,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            response_text = response.split("### Response:")[1].strip()
            
            logger.info(f"💬 Response: {response_text}")
            logger.info("-" * 60)
        
        logger.info("✅ Model testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_original():
    """Compare DPO model with original model (if available)"""
    logger.info("🔄 Comparing DPO model with original model...")
    
    # This would require loading both models and comparing responses
    # Implementation depends on having access to the original model
    logger.info("📝 Comparison feature coming soon!")
    
def main():
    """Main function"""
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level="INFO")
    
    logger.info("🧪 DPO Model Testing Script")
    logger.info("=" * 40)
    
    # Get model path from command line or use default
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./dpo_llm_twin"
    
    # Test the model
    success = test_dpo_model(model_path)
    
    if success:
        logger.info("\n🎉 Model testing completed!")
        logger.info("💡 You can now use this model for inference in your applications")
    else:
        logger.error("\n❌ Model testing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
