# backend/services/agent.py
import json
from typing import Dict, Any

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from config import settings
from services.es_client import es_client
from services.query_safety import QuerySafetyLayer, SafetyStatus
from services.context_manager import ContextManager

query_safety = QuerySafetyLayer()
context_mgr = ContextManager(max_docs=query_safety.max_result_docs)

def _infer_query_type(query: Dict[str, Any]) -> str:
    return "aggregation" if bool(query.get("aggs") or query.get("aggregations")) else "retrieval"

@tool
async def execute_elasticsearch_query(query_json: str) -> str:
    """Executes an Elasticsearch query against the GDELT GKG dataset.
    Input MUST be a valid JSON string representing the Elasticsearch query body.
    """
    try:
        query_dict = json.loads(query_json)
    except Exception as e:
        return f"Error: Invalid JSON provided. {e}"

    validation = query_safety.validate(query_dict)
    if validation.status == SafetyStatus.BLOCKED:
        return f"Error: Query blocked by safety layer. Reason: {validation.reason}. Please revise the query."

    safe_query = validation.query
    query_type = _infer_query_type(safe_query)

    try:
        es_resp = await es_client.search(index=settings.es_index, query=safe_query)
    except Exception as e:
        return f"Error executing query in Elasticsearch: {str(e)}. Check your syntax and schema."

    shaped = context_mgr.shape_results(es_resp, query_type=query_type)
    return json.dumps(shaped, ensure_ascii=False)

def get_agent():
    llm = ChatOpenAI(
        openai_api_base=settings.llm_base_url,
        openai_api_key=settings.llm_api_key,
        model_name=settings.llm_model_name,
        temperature=0,
    )
    # create_react_agent handles the while-loop routing automatically
    return create_react_agent(llm, tools=[execute_elasticsearch_query])