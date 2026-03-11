# backend/routers/chat.py
from fastapi import APIRouter
from datetime import datetime
import json
import logging

from models.schemas import ChatRequest, ChatResponse, QueryMetadata
from services.agent import get_agent
from services.schema_store import get_schema_store

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id
    logger.info("chat_request_received", extra={"session_id": session_id})

    # 1. Retrieve Schema Context
    schema_store = get_schema_store()
    schema_docs = await schema_store.search_schema(request.message, k=8)
    if not schema_docs:
        schema_docs = await schema_store.get_schema_overview(limit=12)
    schema_context = "\n".join(schema_docs) if schema_docs else "(no schema available)"

    # 2. Build the System Prompt
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    system_prompt = f"""You are an OSINT assistant querying the GDELT GKG Elasticsearch index.
Current date and time: {now}

Relevant schema mapping:
{schema_context}

Instructions:
1. You MUST use the `execute_elasticsearch_query` tool to answer the user's question.
2. Write a valid Elasticsearch JSON query using the schema provided.
3. For ranking or grouping ("top 10"), use terms aggregation with size 0 at the top level.
4. If a tool call returns an error or no results, revise your query and call the tool again.
5. After getting successful results, summarize them clearly for the user without mentioning JSON or Elasticsearch.
"""

    # 3. Format messages for LangGraph
    messages = [{"role": "system", "content": system_prompt}]
    for item in request.history:
        messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": request.message})

    # 4. Invoke the ReAct Agent
    agent = get_agent()
    try:
        # ainvoke runs the full generate -> tool call -> observe -> summarize loop
        result = await agent.ainvoke({"messages": messages})
    except Exception as e:
        logger.exception("langgraph_agent_failure")
        return ChatResponse(
            response=f"An error occurred while running the agent: {str(e)}",
            query_metadata=QueryMetadata(
                safety_status="error",
                blocked_reason="agent_runtime_error"
            ),
            session_id=session_id
        )

    # 5. Extract metadata from the LangGraph State trace
    output_messages = result["messages"]
    final_answer = output_messages[-1].content

    for idx, msg in enumerate(output_messages):
            # We only want to log the AI's responses, not the user's prompt or the tool's raw JSON return
            if getattr(msg, "type", "") == "ai" and msg.content:
                logger.info(f"--- AI THINKING STEP {idx} ---")
                logger.info(msg.content)
                logger.info("-----------------------------")

    last_query = {}
    safety_status = "allowed"
    blocked_reason = None

    # Traverse history backward to extract the actual query used and tool outcomes
    for msg in reversed(output_messages):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            try:
                last_query = json.loads(msg.tool_calls[0]["args"].get("query_json", "{}"))
            except Exception:
                pass
            break

    for msg in reversed(output_messages):
        if getattr(msg, "name", None) == "execute_elasticsearch_query":
            if "Error: Query blocked" in str(msg.content):
                safety_status = "blocked"
                blocked_reason = str(msg.content)
            break

    return ChatResponse(
        response=final_answer,
        query_metadata=QueryMetadata(
            es_query=last_query,
            total_hits=None,
            execution_time_ms=None,
            safety_status=safety_status,
            blocked_reason=blocked_reason
        ),
        session_id=session_id
    )