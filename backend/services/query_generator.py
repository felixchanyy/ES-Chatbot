# backend/services/query_generator.py

import json
import re
from datetime import datetime  # 1. Add this import
from typing import Any, Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

class QueryGenerationError(Exception):
    pass

class QueryGenerator:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            openai_api_base=settings.llm_base_url,
            openai_api_key=settings.llm_api_key,
            model_name=settings.llm_model_name,
            temperature=0,
        )
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = Chroma(
            collection_name="gkg_mapping",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db",
        )

    # 2. Update to accept current_time
    def _build_system_prompt(self, schema_context: str, current_time: str) -> str:
        return f"""
    You are an OSINT Assistant. Convert the user's question into a valid Elasticsearch JSON query.

    ### CONTEXT:
    Current Date and Time: {current_time}

    ### APPENDIX A FIELD REFERENCE:
    - Time: 'V21Date'
    - Persons: 'V2Persons.V1Person' (.keyword for aggregations)
    - Organisations: 'V2Orgs.V1Org' (.keyword for aggregations)
    - Locations: 'V2Locations.FullName' (.keyword for aggregations)
    - Country Code: 'V2Locations.CountryCode.keyword'
    - Themes: 'V2EnhancedThemes.V2Theme' (.keyword for aggregations)
    - Tone: 'V15Tone.Tone'
    - Sources: 'V2SrcCmnName.V2SrcCmnName'
    - Title: 'V2ExtrasXML.Title'
    - URL: 'V2DocId'
    - Quotes: 'V21Quotations.Quote'

    ### RULES:
    1. Always use 'V21Date' for date filtering.
    2. For "Top 10" use terms aggregation with '.keyword'.
    3. Set "size": 0 for aggregations.
    4. Target index is always 'gkg'.
    5. Return ONLY valid JSON. No explanations.

    ### SCHEMA CONTEXT:
    {schema_context}
    """.strip()

    def _parse_json(self, text: str) -> Dict[str, Any]:
        if not text:
            raise QueryGenerationError("Empty LLM output.")
        cleaned = text.strip().replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    raise QueryGenerationError(f"Invalid JSON: {e}")
            raise QueryGenerationError("LLM did not return valid JSON.")

    def generate(self, question: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        # 3. Calculate current time
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        docs = self.vectorstore.similarity_search(question, k=6)
        schema_context = "\n".join([d.page_content for d in docs]) if docs else ""

        messages = [
            {"role": "system", "content": self._build_system_prompt(schema_context, now)},
            {"role": "user", "content": question},
        ]

        response = self.llm.invoke(messages)
        return self._parse_json(response.content)