# backend/routers/chat.py

from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse
from services.query_generator import QueryGenerator
from services.query_safety import QuerySafetyLayer
from services.es_client import ESClient
from services.context_manager import ContextManager
from services.response_summariser import ResponseSummariser

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests and return responses."""
    try:
        # Generate Elasticsearch query from user message
        query_generator = QueryGenerator()
        es_query = query_generator.generate(request.message, request.history)
        
        # Validate and sanitize the query
        safety_layer = QuerySafetyLayer()
        validation_result = safety_layer.validate(es_query)
        
        if validation_result.status == "blocked":
            return ChatResponse(
                response="I'm sorry, I can't perform that operation.",
                query_metadata={
                    "es_query": None,
                    "safety_status": "blocked",
                    "blocked_reason": validation_result.reason
                },
                session_id=request.session_id
            )
        
        # Execute the query
        es_client = ESClient()
        es_response = es_client.search(validation_result.query)
        
        # Shape the results
        context_manager = ContextManager()
        shaped_results = context_manager.shape_results(es_response, "aggregation")
        
        # Summarize the results
        summariser = ResponseSummariser()
        summary = summariser.summarize(shaped_results)
        
        return ChatResponse(
            response=summary,
            query_metadata={
                "es_query": validation_result.query,
                "total_hits": es_response.get("hits", {}).get("total", {}).get("value", 0),
                "execution_time_ms": es_response.get("took", 0),
                "safety_status": validation_result.status,
                "blocked_reason": validation_result.reason
            },
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))