# backend/services/query_generator.py

from http import client
import json
import re
from datetime import datetime

from chromadb import HttpClient
from services.langchain_tools import lookup_region
from typing import Any, Dict, List, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

class QueryGenerationError(Exception):
    pass

class QueryGenerator:
    def __init__(self) -> None:
        # LLM setup
        self.llm = ChatOpenAI(
            openai_api_base=settings.llm_base_url,
            openai_api_key=settings.llm_api_key,
            model_name=settings.llm_model_name,
            temperature=0,
        )

        # Vectorstore setup
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        client = HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port
        )

        self.vectorstore = Chroma(
            client=client,
            collection_name="gkg_mapping",
            embedding_function=self.embeddings
        )

        # Tools setup
        self.tools = [lookup_region]
        

        # Prompt setup
        self.prompt = PromptTemplate.from_template(
            """
            You are an OSINT assistant that converts user questions into valid Elasticsearch JSON queries for gdelt data.

            You have access to the following tools:

            {tools}

            Use the following format:

            Question: the user question
            Thought: think about what the user is asking
            Action: the tool to use, must be one of [{tool_names}]
            Action Input: the input to the tool
            Observation: the result of the tool
            ... (this Thought/Action/Observation can repeat up to 3 times)
            Thought: I now know the final answer
            Final Answer: return ONLY valid Elasticsearch JSON

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
            
            Rules:
            - Never explain your reasoning in the Final Answer.
            - Never return markdown.
            - Return ONLY raw JSON.
            - Use V21Date for date filtering.
            - Use .keyword fields for aggregations.
            - For "Top N", use terms aggregation.
            - For aggregations, set "size": 0.
            - Target index is always "gkg".
            - If geographic expansion is required (e.g., "Asia"), use lookup_region tool.
            - Always produce safe, read-only Elasticsearch queries.

            Current Question:
            {input}

            {agent_scratchpad}
            """
        )

        # Agent setup
        self.agent = create_react_agent(
            llm = self.llm,
            tools = self.tools,
            prompt = self.prompt
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            max_iterations=3,
            handle_parsing_errors=True,
        )

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

    async def generate(self, question: str, history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        docs = self.vectorstore.similarity_search(question, k=6)
        schema_context = "\n".join(d.page_content for d in docs)

        enriched_question = f"""
            Current datetime: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

            Schema context:
            {schema_context}

            User question:
            {question}
            """

        response = await self.executor.ainvoke(
            input={"input": enriched_question}
        )
        print(response)
        output = response.get("output")
        
        if not output:
            raise QueryGenerationError("No output from agent.")
        
        return self._parse_json(output)