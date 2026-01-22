"""
FastAPI Backend for Chain of Thought Pipeline.

Exposes the text processing pipeline as a REST API.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.cot.pipeline import CoTPipeline, PipelineConfig
from src.utils.groq_client import GroqClient
from src.ingestors import InputRouter

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Chain of Thought Pipeline API",
    description="Text processing pipeline with NER, Sentiment, Events, Summarization, and more.",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class PipelineRequest(BaseModel):
    text: str = Field(..., description="Input text to process", min_length=1)
    
    # Feature flags
    semantic_clean: bool = Field(False, description="Enable LLM-based semantic cleaning")
    ner: bool = Field(False, description="Enable Named Entity Recognition")
    relationships: bool = Field(True, description="Extract entity relationships (if NER enabled)")
    events: bool = Field(False, description="Enable Event Calendar Extraction")
    sentiment: bool = Field(False, description="Enable Sentiment Analysis")
    summary: bool = Field(False, description="Enable Text Summarization")
    summary_style: str = Field("bullets", description="Summary style (bullets, paragraph, executive, headlines, tldr)")
    translate: bool = Field(False, description="Translate to English first")
    relevancy: bool = Field(False, description="Enable Relevancy Analysis")
    enable_country: bool = Field(False, description="Enable Country/Region Identification")
    topics: Optional[str] = Field(None, description="Comma-separated custom topics for relevancy")
    
    # Advanced
    self_consistency: bool = Field(False, description="Use self-consistency (multiple runs)")
    skip_validation: bool = Field(False, description="Skip validation step")
    skip_domain: bool = Field(False, description="Skip domain detection")
    skip_llm: bool = Field(False, description="Skip all LLM steps (local only)")
    
    # Optimization features
    enable_collaborative_review: bool = Field(False, description="Enable collaborative review (CoLLM) for improved quality")
    enable_hallucination_detection: bool = Field(False, description="Enable hallucination detection for generated content")
    enable_memory_optimization: bool = Field(True, description="Enable intelligent memory/context management")
    token_budget: int = Field(4096, description="Token budget for optimization")

class PipelineResponse(BaseModel):
    status: str
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    duration_ms: int

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Chain of Thought Pipeline API is running. Use /process to process text."}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}

@app.post("/process", response_model=PipelineResponse)
async def process_text(request: PipelineRequest):
    """
    Process text through the Chain of Thought pipeline.
    """
    start_time = time.time()
    
    try:
        # Check API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in server environment")
        
        # Parse custom topics
        custom_topics = None
        if request.topics:
            custom_topics = [t.strip() for t in request.topics.split(",") if t.strip()]
        
        # Configure pipeline
        config = PipelineConfig(
            enable_validation=not request.skip_validation,
            enable_domain_detection=not request.skip_domain and not request.skip_llm,
            enable_semantic_cleaning=request.semantic_clean and not request.skip_llm,
            enable_ner=request.ner and not request.skip_llm,
            enable_relationships=request.relationships,
            enable_events=request.events and not request.skip_llm,
            enable_sentiment=request.sentiment and not request.skip_llm,
            enable_summary=request.summary and not request.skip_llm,
            summary_style=request.summary_style,
            enable_translation=request.translate and not request.skip_llm,
            enable_relevancy=request.relevancy and not request.skip_llm,
            enable_country=request.enable_country and not request.skip_llm,
            relevancy_topics=custom_topics,
            use_self_consistency=request.self_consistency,
            # New optimization features
            enable_collaborative_review=request.enable_collaborative_review and not request.skip_llm,
            enable_hallucination_detection=request.enable_hallucination_detection and not request.skip_llm,
            enable_memory_optimization=request.enable_memory_optimization,
            token_budget=request.token_budget
        )
        
        # Initialize pipeline
        pipeline = CoTPipeline(api_key=api_key, pipeline_config=config)
        
        # Run pipeline
        # Note: If custom topics are needed, we might need to modify CoTPipeline to accept them
        # For now, RelevancyStep uses default topics unless we refactor.
        # Let's handle custom topics in a slightly hacky way for now or assume default topics
        # Ideally, we'd pass runtime args to run(), but run() signature is fixed.
        # We'll stick to default behavior for now to avoid breaking changes to CoTPipeline.runner
        
        results = pipeline.run(request.text)
        
        duration = int((time.time() - start_time) * 1000)
        
        return PipelineResponse(
            status="success",
            results=results,
            metadata=results.get("metadata", {}),
            duration_ms=duration
        )
        
    except Exception as e:
        duration = int((time.time() - start_time) * 1000)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=PipelineResponse)
async def process_file(
    file: UploadFile = File(..., description="PDF or Image file to process"),
    semantic_clean: bool = Form(False),
    ner: bool = Form(False),
    events: bool = Form(False),
    sentiment: bool = Form(False),
    summary: bool = Form(False),
    summary_style: str = Form("bullets"),
    translate: bool = Form(False),
    relevancy: bool = Form(False),
    enable_country: bool = Form(False),
):
    """
    Upload a file (PDF or Image) and process through the pipeline.
    
    Supported formats:
    - PDF (.pdf)
    - Images (.png, .jpg, .jpeg, .webp)
    """
    start_time = time.time()
    
    try:
        # Check API key
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not set")
        
        # Read file content
        content = await file.read()
        
        # Route and extract text
        router = InputRouter(api_key=api_key)
        routed = router.route_bytes(content, file.filename)
        
        # Configure pipeline
        config = PipelineConfig(
            enable_validation=True,
            enable_domain_detection=True,
            enable_semantic_cleaning=semantic_clean,
            enable_ner=ner,
            enable_relationships=True,
            enable_events=events,
            enable_sentiment=sentiment,
            enable_summary=summary,
            summary_style=summary_style,
            enable_translation=translate,
            enable_relevancy=relevancy,
            enable_country=enable_country,
        )
        
        # Run pipeline on extracted text
        pipeline = CoTPipeline(api_key=api_key, pipeline_config=config)
        results = pipeline.run(routed.text)
        
        # Add file metadata to results
        results["file_metadata"] = {
            "filename": file.filename,
            "input_type": routed.input_type.value,
            "source_metadata": routed.metadata,
        }
        
        duration = int((time.time() - start_time) * 1000)
        
        return PipelineResponse(
            status="success",
            results=results,
            metadata=results.get("metadata", {}),
            duration_ms=duration
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
