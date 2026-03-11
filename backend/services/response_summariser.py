"""backend/services/response_summariser.py

Turn shaped ES results into a natural-language answer using the configured LLM.
The summariser now also receives stage_trace so the final answer stays consistent
with the multi-stage pipeline that produced the final result.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class ResponseSummariser:
    """Summarises shaped ES results using the LLM with a robust fallback."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> None:
        self.client = OpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key,
        )
        self.model_name = model_name or settings.llm_model_name
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)

    def summarize(
        self,
        *,
        question: str,
        shaped_results: Dict[str, Any],
        query_type: str,
        stage_trace: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Return a user-facing summary.

        Falls back to a lightweight heuristic summary if the LLM call fails.
        """
        try:
            return self._llm_summary(
                question=question,
                shaped_results=shaped_results,
                query_type=query_type,
                stage_trace=stage_trace or [],
            )
        except Exception:
            logger.exception("LLM summarisation failed; using fallback summary")
            return self._fallback_summary(
                question=question,
                shaped_results=shaped_results,
                query_type=query_type,
                stage_trace=stage_trace or [],
            )

    def _llm_summary(
        self,
        *,
        question: str,
        shaped_results: Dict[str, Any],
        query_type: str,
        stage_trace: List[Dict[str, Any]],
    ) -> str:
        system = (
            "You are an OSINT analyst assistant. "
            "Summarise the provided results for the user's question. "
            "Be factual and thorough. If multiple query attempts were executed, keep the final answer "
            "consistent with the stage trace and final result. "
            "Do not mention Elasticsearch, JSON, or internal implementation details. "
            "If results are empty, say so clearly."
        )

        payload = {
            "question": question,
            "query_type": query_type,
            "stage_trace": stage_trace,
            "results": shaped_results,
        }
        user_content = "Here are the results as JSON:" + json.dumps(payload, ensure_ascii=False)

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=self.max_output_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise LLMError("Empty chat-completions summary")
            return content
        except Exception:
            prompt = f"{system}\n\n{user_content}\n\nAnswer:"
            resp = self.client.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                prompt=prompt,
                max_tokens=self.max_output_tokens,
            )
            text_out = ""
            if getattr(resp, "choices", None):
                text_out = (resp.choices[0].text or "").strip()
            if not text_out:
                raise LLMError("Empty completions summary")
            return text_out

    def _fallback_summary(
        self,
        *,
        question: str,
        shaped_results: Dict[str, Any],
        query_type: str,
        stage_trace: List[Dict[str, Any]],
    ) -> str:
        if not isinstance(shaped_results, dict):
            return "I couldn't summarise the results, but the query executed successfully."

        total = shaped_results.get("total_hits")
        attempted_multiple = len(stage_trace) > 1

        if query_type == "aggregation":
            aggs = shaped_results.get("aggregations")
            if isinstance(aggs, dict) and aggs:
                first_name = next(iter(aggs.keys()))
                buckets = aggs.get(first_name)
                if isinstance(buckets, list) and buckets:
                    top = buckets[:10]
                    lines = [f"Top results for '{first_name}':"]
                    for i, bucket in enumerate(top, 1):
                        key = bucket.get("key")
                        cnt = bucket.get("doc_count")
                        lines.append(f"{i}. {key} — {cnt}")
                    if total is not None:
                        lines.append(f"(Matched about {total} articles in total.)")
                    return "\n".join(lines)

            if attempted_multiple:
                return (
                    "I tried a few query strategies, but none produced aggregation buckets worth summarising. "
                    "Try broadening the time range or simplifying the entity/keyword requested."
                )
            return "I ran the analysis, but there were no aggregation buckets to summarise."

        docs = shaped_results.get("documents")
        if isinstance(docs, list) and docs:
            lines = ["Here are a few matching articles:"]
            for d in docs[:5]:
                if not isinstance(d, dict):
                    continue
                v2extras = d.get("V2ExtrasXML")
                title = v2extras.get("Title") if isinstance(v2extras, dict) else d.get("V2ExtrasXML.Title")
                url = d.get("V2DocId") or d.get("DocumentIdentifier") or d.get("url")
                date = d.get("V21Date") or d.get("date")
                parts = [p for p in [date, title, url] if p]
                if parts:
                    lines.append("- " + " | ".join(str(p) for p in parts))
            if total is not None:
                lines.append(f"(Matched about {total} articles in total.)")
            return "\n".join(lines)

        if isinstance(total, int) and total > 0:
            if attempted_multiple:
                return (
                    f"I tried multiple query strategies and found about {total} matching articles, "
                    "but there were no document previews to show in the final result."
                )
            return f"The query matched about {total} articles, but there were no document previews to summarise."

        if attempted_multiple:
            return (
                "I tried a few query strategies, but none returned useful results. "
                "Try broadening the time range or using a different keyword."
            )
        return "No matching articles were found for that question. Try broadening the time range or using a different keyword."