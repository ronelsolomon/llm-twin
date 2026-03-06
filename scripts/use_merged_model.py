#!/usr/bin/env python3
"""
Example script showing how to use the merged DPO model
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger

def load_merged_model(model_path: str = "./dpo_llm_twin_merged"):
    """Load the merged DPO model for inference"""
    
    logger.info(f"🤖 Loading merged model from: {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    logger.info("✅ Model loaded successfully")
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """Generate response from the model"""
    
    # Format prompt with Alpaca template
    alpaca_template = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.
### Instruction:
{}
### Response:
"""
    
    formatted_prompt = alpaca_template.format(prompt)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part
    response_text = response.split("### Response:")[1].strip()
    
    return response_text

def main():
    """Main function"""
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level="INFO")
    
    logger.info("🚀 Using Merged DPO Model")
    logger.info("=" * 40)
    
    try:
        # Load model
        model, tokenizer = load_merged_model()
        
        # Test prompts
        test_prompts = [
            "What are your main areas of expertise and experience?",
            "Can you describe your approach to software development?",
            "What technologies do you work with most frequently?",
            "How do you approach problem-solving in your projects?"
        ]
        
        logger.info("🧪 Testing model with sample prompts:")
        logger.info("=" * 60)
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"\n📝 Test {i}: {prompt}")
            
            # Generate response
            response = generate_response(model, tokenizer, prompt)
            
            logger.info(f"💬 Response: {response}")
            logger.info("-" * 60)
        
        logger.info("✅ Model testing completed!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
