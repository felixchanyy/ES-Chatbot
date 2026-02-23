from fastapi import APIRouter, HTTPException

from models.schemas import ChatRequest, ChatResponse, QueryMetadata
from services.query_generator import QueryGenerator, QueryGenerationError

router = APIRouter()
query_gen = QueryGenerator()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Milestone 2:
    Generate ES query JSON and return it.
    No execution yet.
    """

    try:
        es_query = query_gen.generate(request.message, request.history)

        return ChatResponse(
            response="Elasticsearch query successfully generated.",
            query_metadata=QueryMetadata(
                es_query=es_query,
                total_hits=None,
                execution_time_ms=None,
                safety_status="allowed",  # Required by schema
                blocked_reason=None
            ),
            session_id=request.session_id,
        )

    except QueryGenerationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))