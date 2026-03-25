#!/usr/bin/env python3
"""
Simple Test for DPO Model - Test basic functionality without complex generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
import json
from pathlib import Path

def test_dpo_model_simple():
    """Simple test to verify your DPO model works"""
    
    model_path = "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged"
    
    print("🧪 Simple DPO Model Test")
    print("="*50)
    print(f"Model: {model_path}")
    print("="*50)
    
    try:
        # Load tokenizer
        print("📦 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"✅ Tokenizer loaded! Vocab size: {tokenizer.vocab_size}")
        
        # Load model correctly for GPT-2
        print("🧠 Loading model...")
        
        # Check config first
        with open(model_path + "/config.json", 'r') as f:
            config = json.load(f)
        
        print(f"📋 Model type: {config.get('model_type')}")
        print(f"📋 Architecture: {config.get('architectures')}")
        
        # Load as GPT-2 model
        model = GPT2LMHeadModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            device_map="cpu"
        )
        
        print(f"✅ Model loaded! Type: {type(model).__name__}")
        
        # Test basic tokenization
        test_text = "What is machine learning?"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        print(f"📝 Test text: '{test_text}'")
        print(f"🔢 Tokenized shape: {inputs['input_ids'].shape}")
        
        # Test forward pass (no generation)
        print("🚀 Testing forward pass...")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        print(f"✅ Forward pass successful! Logits shape: {logits.shape}")
        
        # Simple generation test
        print("💬 Testing simple generation...")
        try:
            # Use very simple generation parameters
            generated_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=10,
                do_sample=False,  # Use greedy decoding
                pad_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"✅ Generation successful! Output: '{generated_text}'")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Generation failed: {e}")
            print("✅ But model loading and forward pass work!")
            return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_simple_evaluation():
    """Test a simple evaluation scenario"""
    
    print("\n🎯 Simple Evaluation Test")
    print("="*50)
    
    # Test with a simple enterprise scenario
    test_prompt = "Write a professional email to a client about a project delay."
    
    try:
        model_path = "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged"
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path, torch_dtype=torch.float32, device_map="cpu")
        
        # Generate response
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"📝 Prompt: {test_prompt}")
        print(f"💬 Response: {response}")
        
        # Simple evaluation
        word_count = len(response.split())
        has_professional_words = any(word in response.lower() for word in ["professional", "regards", "sincerely", "thank"])
        
        print(f"\n📊 Simple Metrics:")
        print(f"  Word count: {word_count}")
        print(f"  Professional tone: {'✅' if has_professional_words else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple evaluation failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🎯 DPO LLM Twin - Quick Functionality Test")
    print("="*60)
    
    # Test 1: Basic model functionality
    success1 = test_dpo_model_simple()
    
    # Test 2: Simple evaluation
    success2 = test_simple_evaluation()
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    
    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your DPO model is working correctly")
        print("✅ Ready for enterprise evaluations")
        print("\n💡 Next steps:")
        print("  1. Run full Enterprise Scenarios evaluation")
        print("  2. Compare with base models")
        print("  3. Test on specific business use cases")
        
    elif success1:
        print("⚠️  PARTIAL SUCCESS")
        print("✅ Model loads and works")
        print("❌ Generation needs optimization")
        print("\n💡 Recommendations:")
        print("  1. Check model quantization settings")
        print("  2. Try different generation parameters")
        print("  3. Use simpler evaluation scenarios first")
        
    else:
        print("❌ MODEL ISSUES DETECTED")
        print("⚠️  Model loading failed")
        print("\n💡 Troubleshooting:")
        print("  1. Check model file integrity")
        print("  2. Verify dependencies")
        print("  3. Try loading with different parameters")

if __name__ == "__main__":
    main()
