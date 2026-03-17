from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from config import settings


class ResponseSummariser:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model_name,
            temperature=0.2,
            timeout=settings.llm_timeout_seconds,
        )

    async def summarize(
        self,
        *,
        question: str,
        shaped_results: Dict[str, Any],
        stage_trace: List[Dict[str, Any]],
    ) -> str:
        prompt = {
            "question": question,
            "stage_trace": stage_trace,
            "results": shaped_results,
            "instructions": [
                "Answer the user directly.",
                "Use only the final validated results.",
                "Do not mention Elasticsearch, JSON, tools, or hidden reasoning.",
                "If the system could not fully satisfy the requested count, say so clearly.",
            ],
        }
        try:
            response = await asyncio.to_thread(
                self.llm.invoke,
                [
                    {
                        "role": "system",
                        "content": "You are an OSINT analyst assistant. Write a concise factual answer.",
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
            )
            content = getattr(response, "content", "")
            if isinstance(content, list):
                content = "\n".join(str(part) for part in content)
            text = str(content).strip()
            if text:
                return text
        except Exception:
            pass
        return self._fallback_summary(question=question, shaped_results=shaped_results)

    def _fallback_summary(self, *, question: str, shaped_results: Dict[str, Any]) -> str:
        validation = shaped_results.get("validation") or {}
        aggs = shaped_results.get("aggregations") or {}

        if isinstance(aggs, dict) and aggs:
            first_name = next(iter(aggs.keys()))
            buckets = aggs.get(first_name) or []
            if isinstance(buckets, list) and buckets:
                lines = []

                if validation and not validation.get("passed", True):
                    lines.append(
                        f"I could only validate {validation.get('valid_count', 0)} out of "
                        f"{validation.get('requested_count', 0)} requested results."
                    )
                    lines.append("Here are the validated results I found:")

                else:
                    lines.append(f"Results for: {question}")

                for idx, bucket in enumerate(buckets[:10], start=1):
                    lines.append(f"{idx}. {bucket.get('key')} — {bucket.get('doc_count')}")
                return "\n".join(lines)

        docs = shaped_results.get("documents") or []
        if docs:
            return f"I found {len(docs)} matching documents for: {question}"

        return "I could not find validated results that fully satisfy the request."