#!/usr/bin/env python3
"""
LLM Twin Inference Pipeline
A comprehensive inference system for the DPO fine-tuned LLM twin models.
"""
import torch
import json
import time
from typing import Dict, List, Any, Optional, Union, AsyncGenerator
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from loguru import logger
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    GenerationConfig,
    TextStreamer
)
from concurrent.futures import ThreadPoolExecutor
import asyncio

from src.config import settings


@dataclass
class InferenceConfig:
    """Configuration for model inference"""
    model_path: str = "./dpo_llm_twin_merged"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    device: str = "auto"
    use_cache: bool = True
    batch_size: int = 4
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class GenerationRequest:
    """Single generation request"""
    prompt: str
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResponse:
    """Generation response with metadata"""
    generated_text: str
    prompt: str
    generation_time: float
    token_count: int
    tokens_per_second: float
    config_used: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMTwinInferencePipeline:
    """Main inference pipeline for LLM Twin models"""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.model = None
        self.tokenizer = None
        self.generation_config = None
        self.device = self.config.device
        self.is_loaded = False
        
        logger.info(f"🤖 LLM Twin Inference Pipeline initialized")
        logger.info(f"📱 Device: {self.device}")
        logger.info(f"🔧 Model path: {self.config.model_path}")
    
    def load_model(self) -> bool:
        """Load the model and tokenizer"""
        try:
            logger.info(f"📦 Loading model from: {self.config.model_path}")
            
            # Check if model exists
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at: {self.config.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with appropriate settings
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto" if self.device != "cpu" else None,
                "low_cpu_mem_usage": True,
            }
            
            if self.device == "cpu":
                model_kwargs["device_map"] = None
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )
            
            # Create generation config
            self.generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=self.config.use_cache
            )
            
            self.is_loaded = True
            logger.info("✅ Model and tokenizer loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.is_loaded = False
            return False
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using Ronel AI Twin template (consistent with training)"""
        alpaca_template = """You are my AI twin.

Your name is Ronel Solomon.

Speak in first person as Ronel.

You're a senior ML/AI engineer focused on LLM security, MLOps, distributed systems, and FastAPI.

If the user asks who you are, say: 'I'm Ronel.'

Stay technical, concise, and avoid emojis unless explicitly requested.

### Instruction:
{}
### Response:
"""
        return alpaca_template.format(prompt)
    
    def _extract_response(self, full_response: str) -> str:
        """Extract only the response part from the full generation"""
        if "### Response:" in full_response:
            return full_response.split("### Response:")[1].strip()
        return full_response.strip()
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response for a single request"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Override config with request-specific settings
        config = self.config
        max_tokens = request.max_new_tokens or config.max_new_tokens
        temperature = request.temperature or config.temperature
        top_p = request.top_p or config.top_p
        
        # Format prompt
        formatted_prompt = self._format_prompt(request.prompt)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Update generation config for this request
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=config.top_k,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=config.use_cache
        )
        
        try:
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config
                )
            
            # Decode and extract response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = self._extract_response(full_response)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            token_count = len(self.tokenizer.encode(response_text))
            tokens_per_second = token_count / generation_time if generation_time > 0 else 0
            
            return GenerationResponse(
                generated_text=response_text,
                prompt=request.prompt,
                generation_time=generation_time,
                token_count=token_count,
                tokens_per_second=tokens_per_second,
                config_used=gen_config.to_dict(),
                user_id=request.user_id,
                session_id=request.session_id,
                metadata=request.metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            raise
    
    def generate_batch(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """Generate responses for multiple requests in batch"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info(f"🔄 Processing batch of {len(requests)} requests")
        
        responses = []
        batch_size = self.config.batch_size
        
        # Process in batches
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            
            # Format prompts
            formatted_prompts = [self._format_prompt(req.prompt) for req in batch]
            
            # Tokenize batch
            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate for batch
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )
            
            generation_time = time.time() - start_time
            
            # Process each output in the batch
            for j, (request, output) in enumerate(zip(batch, outputs)):
                full_response = self.tokenizer.decode(output, skip_special_tokens=True)
                response_text = self._extract_response(full_response)
                
                token_count = len(self.tokenizer.encode(response_text))
                tokens_per_second = token_count / generation_time if generation_time > 0 else 0
                
                response = GenerationResponse(
                    generated_text=response_text,
                    prompt=request.prompt,
                    generation_time=generation_time / len(batch),  # Approximate per-request time
                    token_count=token_count,
                    tokens_per_second=tokens_per_second,
                    config_used=self.generation_config.to_dict(),
                    user_id=request.user_id,
                    session_id=request.session_id,
                    metadata=request.metadata
                )
                responses.append(response)
        
        logger.info(f"✅ Batch processing completed: {len(responses)} responses")
        return responses
    
    async def generate_stream(self, request: GenerationRequest) -> AsyncGenerator[str, None]:
        """Generate response with streaming"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Format prompt
        formatted_prompt = self._format_prompt(request.prompt)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Create streamer
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generation config for streaming
        gen_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens or self.config.max_new_tokens,
            temperature=request.temperature or self.config.temperature,
            top_p=request.top_p or self.config.top_p,
            top_k=self.config.top_k,
            do_sample=self.config.do_sample,
            repetition_penalty=self.config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=self.config.use_cache
        )
        
        # Generate in a separate thread to allow streaming
        def generate_in_thread():
            with torch.no_grad():
                return self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                    streamer=streamer
                )
        
        # Run generation in background thread
        with ThreadPoolExecutor() as executor:
            future = executor.submit(generate_in_thread)
            
            # Yield tokens as they're generated
            for token in streamer:
                yield token
            
            # Wait for generation to complete
            future.result()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"error": "Model not loaded"}
        
        model_info = {
            "model_path": self.config.model_path,
            "device": self.device,
            "model_type": type(self.model).__name__,
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": getattr(self.model.config, 'max_position_embeddings', 'N/A'),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "is_loaded": self.is_loaded,
            "config": {f.name: getattr(self.config, f.name) for f in fields(InferenceConfig)}
        }
        
        return model_info
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the inference pipeline"""
        health_status = {
            "status": "healthy",
            "model_loaded": self.is_loaded,
            "device_available": torch.cuda.is_available() or torch.backends.mps.is_available() or True,
            "memory_usage": self._get_memory_usage(),
            "timestamp": time.time()
        }
        
        if not self.is_loaded:
            health_status["status"] = "unhealthy"
            health_status["error"] = "Model not loaded"
        
        return health_status
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            memory_info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        
        return memory_info
    
    def unload_model(self):
        """Unload model and tokenizer to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        logger.info("🗑️ Model unloaded and memory cleared")


# Convenience functions for quick usage
def create_inference_pipeline(model_path: str = "./dpo_llm_twin_merged") -> LLMTwinInferencePipeline:
    """Create and initialize inference pipeline with default settings"""
    config = InferenceConfig(model_path=model_path)
    pipeline = LLMTwinInferencePipeline(config)
    pipeline.load_model()
    return pipeline


def quick_generate(prompt: str, model_path: str = "./dpo_llm_twin_merged") -> str:
    """Quick generation function for simple use cases"""
    pipeline = create_inference_pipeline(model_path)
    request = GenerationRequest(prompt=prompt)
    response = pipeline.generate(request)
    pipeline.unload_model()
    return response.generated_text
