#!/usr/bin/env python3
"""
LLM Twin Inference Server
A FastAPI-based server for the LLM Twin inference pipeline.
Provides REST API endpoints for text generation.
"""
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import uuid
from contextlib import asynccontextmanager
import json
import sys

from src.inference_pipeline import (
    LLMTwinInferencePipeline,
    InferenceConfig,
    GenerationRequest,
    GenerationResponse
)
from loguru import logger


# Pydantic models for API
class GenerationRequestModel(BaseModel):
    prompt: str = Field(..., description="The prompt to generate text for")
    max_new_tokens: Optional[int] = Field(256, description="Maximum new tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Generation temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter")
    user_id: Optional[str] = Field(None, description="User ID for tracking")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class BatchGenerationRequestModel(BaseModel):
    requests: List[GenerationRequestModel] = Field(..., description="List of generation requests")


class GenerationResponseModel(BaseModel):
    generated_text: str
    prompt: str
    generation_time: float
    token_count: int
    tokens_per_second: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelInfoModel(BaseModel):
    model_path: str
    device: str
    model_type: str
    vocab_size: int
    max_position_embeddings: int
    num_parameters: int
    is_loaded: bool
    config: Dict[str, Any]


class HealthCheckModel(BaseModel):
    status: str
    model_loaded: bool
    device_available: bool
    memory_usage: Dict[str, float]
    timestamp: float


# Global pipeline instance
pipeline: Optional[LLMTwinInferencePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the inference pipeline"""
    global pipeline
    
    # Startup
    logger.info("🚀 Starting LLM Twin Inference Server")
    
    try:
        # Initialize pipeline
        config = InferenceConfig(
            model_path="./dpo_llm_twin_merged",
            batch_size=4,
            device="auto"
        )
        
        pipeline = LLMTwinInferencePipeline(config)
        success = pipeline.load_model()
        
        if not success:
            logger.error("❌ Failed to load model on startup")
            raise RuntimeError("Model loading failed")
        
        logger.info("✅ Inference server started successfully")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down inference server")
    if pipeline:
        pipeline.unload_model()
    logger.info("✅ Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LLM Twin Inference API",
    description="REST API for LLM Twin text generation",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Twin Inference API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check() -> HealthCheckModel:
    """Health check endpoint"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    health = pipeline.health_check()
    return HealthCheckModel(**health)


@app.get("/model/info")
async def get_model_info() -> ModelInfoModel:
    """Get model information"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    info = pipeline.get_model_info()
    return ModelInfoModel(**info)


@app.post("/generate")
async def generate_text(request: GenerationRequestModel) -> GenerationResponseModel:
    """Generate text for a single prompt"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Convert API request to internal request
        gen_request = GenerationRequest(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            user_id=request.user_id,
            session_id=request.session_id,
            metadata=request.metadata
        )
        
        # Generate response
        response = pipeline.generate(gen_request)
        
        # Convert to API response
        return GenerationResponseModel(
            generated_text=response.generated_text,
            prompt=response.prompt,
            generation_time=response.generation_time,
            token_count=response.token_count,
            tokens_per_second=response.tokens_per_second,
            user_id=response.user_id,
            session_id=response.session_id,
            metadata=response.metadata
        )
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/batch")
async def generate_batch(request: BatchGenerationRequestModel) -> List[GenerationResponseModel]:
    """Generate text for multiple prompts in batch"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        # Convert API requests to internal requests
        gen_requests = []
        for req in request.requests:
            gen_request = GenerationRequest(
                prompt=req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                user_id=req.user_id,
                session_id=req.session_id,
                metadata=req.metadata
            )
            gen_requests.append(gen_request)
        
        # Generate batch responses
        responses = pipeline.generate_batch(gen_requests)
        
        # Convert to API responses
        api_responses = []
        for response in responses:
            api_response = GenerationResponseModel(
                generated_text=response.generated_text,
                prompt=response.prompt,
                generation_time=response.generation_time,
                token_count=response.token_count,
                tokens_per_second=response.tokens_per_second,
                user_id=response.user_id,
                session_id=response.session_id,
                metadata=response.metadata
            )
            api_responses.append(api_response)
        
        return api_responses
        
    except Exception as e:
        logger.error(f"❌ Batch generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate/stream")
async def generate_stream(prompt: str, max_new_tokens: int = 256, temperature: float = 0.7):
    """Generate text with streaming (Server-Sent Events)"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    async def stream_generator():
        try:
            request = GenerationRequest(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            async for token in pipeline.generate_stream(request):
                yield f"data: {json.dumps({'token': token})}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"❌ Streaming failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """Reload the model (useful for model updates)"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    def reload():
        try:
            logger.info("🔄 Reloading model...")
            pipeline.unload_model()
            success = pipeline.load_model()
            
            if success:
                logger.info("✅ Model reloaded successfully")
            else:
                logger.error("❌ Model reload failed")
                
        except Exception as e:
            logger.error(f"❌ Model reload error: {e}")
    
    background_tasks.add_task(reload)
    return {"message": "Model reload initiated"}


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    health = pipeline.health_check()
    model_info = pipeline.get_model_info()
    
    return {
        "server_status": "running",
        "model_loaded": health["model_loaded"],
        "device": model_info["device"],
        "memory_usage": health["memory_usage"],
        "num_parameters": model_info["num_parameters"],
        "uptime": time.time() - health["timestamp"]
    }


def main():
    """Run the inference server"""
    logger.remove()
    logger.add(
        "logs/inference_server.log",
        rotation="10 MB",
        retention="1 week",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    logger.info("🚀 Starting LLM Twin Inference Server")
    
    # Run server
    uvicorn.run(
        "tools.inference_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    import sys
    main()
