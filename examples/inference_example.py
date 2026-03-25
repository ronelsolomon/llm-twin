#!/usr/bin/env python3
"""
Example usage of the LLM Twin Inference Pipeline
Demonstrates various ways to use the inference system.
"""
import asyncio
import json
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference_pipeline import (
    LLMTwinInferencePipeline,
    InferenceConfig,
    GenerationRequest,
    create_inference_pipeline,
    quick_generate
)
from loguru import logger


def example_basic_usage():
    """Example: Basic single generation"""
    logger.info("🚀 Example 1: Basic Single Generation")
    logger.info("=" * 50)
    
    try:
        # Create pipeline with default settings
        pipeline = create_inference_pipeline()
        
        # Generate response
        request = GenerationRequest(
            prompt="What are your main areas of expertise and experience?",
            temperature=0.7,
            max_new_tokens=256
        )
        
        response = pipeline.generate(request)
        
        logger.info(f"📝 Prompt: {response.prompt}")
        logger.info(f"💬 Response: {response.generated_text}")
        logger.info(f"⏱️  Generation time: {response.generation_time:.2f}s")
        logger.info(f"🔢 Tokens: {response.token_count}")
        logger.info(f"⚡ Tokens/sec: {response.tokens_per_second:.2f}")
        
        # Cleanup
        pipeline.unload_model()
        
    except Exception as e:
        logger.error(f"❌ Error in basic usage: {e}")


def example_batch_generation():
    """Example: Batch generation for multiple prompts"""
    logger.info("\n🚀 Example 2: Batch Generation")
    logger.info("=" * 50)
    
    try:
        # Create pipeline with custom config
        config = InferenceConfig(
            model_path="./dpo_llm_twin_merged",
            batch_size=2,
            temperature=0.8,
            max_new_tokens=200
        )
        
        pipeline = LLMTwinInferencePipeline(config)
        pipeline.load_model()
        
        # Create multiple requests
        requests = [
            GenerationRequest(
                prompt="Can you describe your approach to software development?",
                user_id="user1"
            ),
            GenerationRequest(
                prompt="What technologies do you work with most frequently?",
                user_id="user1"
            ),
            GenerationRequest(
                prompt="How do you approach problem-solving in your projects?",
                user_id="user2"
            ),
            GenerationRequest(
                prompt="What's your experience with machine learning and AI?",
                user_id="user2"
            )
        ]
        
        # Generate batch responses
        responses = pipeline.generate_batch(requests)
        
        # Display results
        for i, response in enumerate(responses, 1):
            logger.info(f"\n--- Response {i} (User: {response.user_id}) ---")
            logger.info(f"📝 Prompt: {response.prompt}")
            logger.info(f"💬 Response: {response.generated_text}")
            logger.info(f"⏱️  Time: {response.generation_time:.2f}s")
        
        # Cleanup
        pipeline.unload_model()
        
    except Exception as e:
        logger.error(f"❌ Error in batch generation: {e}")


async def example_streaming_generation():
    """Example: Streaming generation"""
    logger.info("\n🚀 Example 3: Streaming Generation")
    logger.info("=" * 50)
    
    try:
        pipeline = create_inference_pipeline()
        
        request = GenerationRequest(
            prompt="Tell me about your experience with large language models and AI development.",
            max_new_tokens=300,
            temperature=0.7
        )
        
        logger.info("📝 Prompt: " + request.prompt)
        logger.info("💬 Streaming response: ", end="")
        
        # Stream generation
        async for token in pipeline.generate_stream(request):
            print(token, end="", flush=True)
        
        print("\n")
        
        # Cleanup
        pipeline.unload_model()
        
    except Exception as e:
        logger.error(f"❌ Error in streaming generation: {e}")


def example_model_information():
    """Example: Get model information and health check"""
    logger.info("\n🚀 Example 4: Model Information & Health Check")
    logger.info("=" * 50)
    
    try:
        pipeline = create_inference_pipeline()
        
        # Get model info
        model_info = pipeline.get_model_info()
        logger.info("📊 Model Information:")
        for key, value in model_info.items():
            if key != "config":
                logger.info(f"  {key}: {value}")
        
        # Health check
        health = pipeline.health_check()
        logger.info("\n🏥 Health Check:")
        for key, value in health.items():
            logger.info(f"  {key}: {value}")
        
        # Cleanup
        pipeline.unload_model()
        
    except Exception as e:
        logger.error(f"❌ Error getting model info: {e}")


def example_quick_generate():
    """Example: Quick generation function"""
    logger.info("\n🚀 Example 5: Quick Generation Function")
    logger.info("=" * 50)
    
    try:
        prompts = [
            "What's your favorite programming language and why?",
            "How do you stay updated with the latest technology trends?",
            "What advice would you give to aspiring developers?"
        ]
        
        for prompt in prompts:
            logger.info(f"📝 Prompt: {prompt}")
            response = quick_generate(prompt)
            logger.info(f"💬 Response: {response}")
            logger.info("-" * 30)
        
    except Exception as e:
        logger.error(f"❌ Error in quick generation: {e}")


def example_performance_benchmark():
    """Example: Performance benchmarking"""
    logger.info("\n🚀 Example 6: Performance Benchmark")
    logger.info("=" * 50)
    
    try:
        pipeline = create_inference_pipeline()
        
        # Test prompts
        test_prompts = [
            "What are your main areas of expertise?",
            "Describe your software development approach.",
            "What technologies do you use frequently?",
            "How do you solve complex problems?",
            "What's your experience with AI/ML?"
        ]
        
        # Run benchmark
        total_time = 0
        total_tokens = 0
        
        logger.info("🏃 Running benchmark...")
        
        for i, prompt in enumerate(test_prompts, 1):
            start_time = time.time()
            
            request = GenerationRequest(
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.7
            )
            
            response = pipeline.generate(request)
            
            total_time += response.generation_time
            total_tokens += response.token_count
            
            logger.info(f"Test {i}: {response.generation_time:.2f}s, {response.token_count} tokens")
        
        # Calculate averages
        avg_time = total_time / len(test_prompts)
        avg_tokens = total_tokens / len(test_prompts)
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        
        logger.info(f"\n📊 Benchmark Results:")
        logger.info(f"  Average time per generation: {avg_time:.2f}s")
        logger.info(f"  Average tokens per generation: {avg_tokens:.0f}")
        logger.info(f"  Average tokens per second: {avg_tps:.2f}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Total tokens: {total_tokens}")
        
        # Cleanup
        pipeline.unload_model()
        
    except Exception as e:
        logger.error(f"❌ Error in benchmark: {e}")


def example_custom_configuration():
    """Example: Custom configuration"""
    logger.info("\n🚀 Example 7: Custom Configuration")
    logger.info("=" * 50)
    
    try:
        # Custom configuration for different use cases
        configs = {
            "creative": InferenceConfig(
                temperature=0.9,
                top_p=0.95,
                top_k=50,
                max_new_tokens=300,
                repetition_penalty=1.1
            ),
            "focused": InferenceConfig(
                temperature=0.3,
                top_p=0.8,
                top_k=40,
                max_new_tokens=200,
                repetition_penalty=1.2
            ),
            "balanced": InferenceConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_new_tokens=256,
                repetition_penalty=1.1
            )
        }
        
        prompt = "What are your thoughts on the future of artificial intelligence?"
        
        for config_name, config in configs.items():
            logger.info(f"\n--- {config_name.title()} Configuration ---")
            
            pipeline = LLMTwinInferencePipeline(config)
            pipeline.load_model()
            
            request = GenerationRequest(
                prompt=prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature
            )
            
            response = pipeline.generate(request)
            
            logger.info(f"💬 Response ({config_name}): {response.generated_text[:150]}...")
            logger.info(f"⚡ Speed: {response.tokens_per_second:.2f} tokens/sec")
            
            pipeline.unload_model()
        
    except Exception as e:
        logger.error(f"❌ Error in custom configuration: {e}")


def main():
    """Run all examples"""
    logger.remove()
    logger.add(
        sys.stdout, 
        level="INFO", 
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    logger.info("🎯 LLM Twin Inference Pipeline Examples")
    logger.info("=" * 60)
    
    # Check if model exists
    model_path = Path("./dpo_llm_twin_merged")
    if not model_path.exists():
        logger.error(f"❌ Model not found at {model_path}")
        logger.error("Please ensure the model is available before running examples")
        return
    
    try:
        # Run examples
        example_basic_usage()
        example_batch_generation()
        asyncio.run(example_streaming_generation())
        example_model_information()
        example_quick_generate()
        example_performance_benchmark()
        example_custom_configuration()
        
        logger.info("\n🎉 All examples completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Examples interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
