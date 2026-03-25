#!/usr/bin/env python3
"""
Simple test script for LLM Twin Inference Pipeline
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.inference_pipeline import (
    LLMTwinInferencePipeline,
    InferenceConfig,
    GenerationRequest,
    create_inference_pipeline
)

def test_basic_generation():
    """Test basic text generation"""
    print("🤖 Testing LLM Twin Inference Pipeline")
    print("=" * 50)
    
    try:
        # Create pipeline
        pipeline = create_inference_pipeline()
        
        # Test prompts
        test_prompts = [
            "What are your main areas of expertise and experience?",
            "Can you describe your approach to software development?",
            "What technologies do you work with most frequently?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n--- Test {i} ---")
            print(f"📝 Prompt: {prompt}")
            
            # Generate response
            request = GenerationRequest(
                prompt=prompt,
                max_new_tokens=200,
                temperature=0.7
            )
            
            response = pipeline.generate(request)
            
            print(f"💬 Response: {response.generated_text}")
            print(f"⏱️  Time: {response.generation_time:.2f}s")
            print(f"🔢 Tokens: {response.token_count}")
            print(f"⚡ Speed: {response.tokens_per_second:.2f} tokens/sec")
        
        # Cleanup
        pipeline.unload_model()
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_generation()
