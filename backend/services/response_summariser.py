"""backend/services/response_summariser.py

Milestone 4:
Turn shaped ES results into a natural-language answer using the local LLM.

Compatibility note:
Different OpenAI-compatible servers expose different endpoints:
- /v1/chat/completions (Chat Completions)
- /v1/completions (Text Completions)

MiniMax-M2.1-AWQ is commonly served via vLLM/LM Studio/etc with an OpenAI-compatible API.
This summariser will try Chat Completions first, then fall back to Text Completions.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from openai import OpenAI

from config import settings


class LLMError(RuntimeError):
    pass


class ResponseSummariser:
    """Summarises shaped ES results using the LLM."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 350,
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
    ) -> str:
        """Return a user-facing summary.

        Falls back to a lightweight heuristic summary if the LLM call fails.
        """
        try:
            return self._llm_summary(question=question, shaped_results=shaped_results, query_type=query_type)
        except Exception:
            return self._fallback_summary(question=question, shaped_results=shaped_results, query_type=query_type)

    def _llm_summary(self, *, question: str, shaped_results: Dict[str, Any], query_type: str) -> str:
        system = (
            "You are an OSINT analyst assistant. "
            "Summarise the provided results for the user's question. "
            "Be concise and factual. "
            "Do not mention Elasticsearch, JSON, or internal implementation details. "
            "If results are empty, say so and suggest how to broaden the query (e.g., wider time range)."
        )

        payload = {
            "question": question,
            "query_type": query_type,
            "results": shaped_results,
        }
        user_content = "Here are the results as JSON:" + json.dumps(payload, ensure_ascii=False)

        # 1) Prefer Chat Completions (works on most OpenAI-compatible servers)
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                # Some servers ignore this; harmless where unsupported.
                max_tokens=self.max_output_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                raise LLMError("Empty chat-completions summary")
            return content
        except Exception:
            # 2) Fall back to Text Completions (/v1/completions) for servers that don't implement chat
            prompt = f"{system}\n\n{user_content}\n\nAnswer:"
            resp = self.client.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                prompt=prompt,
                max_tokens=self.max_output_tokens,
            )
            # openai SDK returns choices[].text for completions
            text_out = ""
            if getattr(resp, "choices", None):
                text_out = (resp.choices[0].text or "").strip()
            if not text_out:
                raise LLMError("Empty completions summary")
            return text_out

    def _fallback_summary(self, *, question: str, shaped_results: Dict[str, Any], query_type: str) -> str:
        # Minimal heuristic: handle common shapes.
        if not isinstance(shaped_results, dict):
            return "I couldn't summarise the results, but the query executed successfully."

        total = shaped_results.get("total_hits")

        if query_type == "aggregation":
            aggs = shaped_results.get("aggregations")
            if isinstance(aggs, dict) and aggs:
                # Pick first agg
                first_name = next(iter(aggs.keys()))
                buckets = aggs.get(first_name)
                if isinstance(buckets, list) and buckets:
                    top = buckets[:10]
                    lines = [f"Top results for '{first_name}':"]
                    for i, b in enumerate(top, 1):
                        key = b.get("key")
                        cnt = b.get("doc_count")
                        lines.append(f"{i}. {key} — {cnt}")
                    if total is not None:
                        lines.append(f"(Matched about {total} articles in total.)")
                    return "\n".join(lines)
            return "I ran the analysis, but there were no aggregation buckets to summarise."

        # retrieval
        docs = shaped_results.get("documents")
        if isinstance(docs, list) and docs:
            lines = ["Here are a few matching articles:"]
            for d in docs[:5]:
                if not isinstance(d, dict):
                    continue
                title = None
                v2extras = d.get("V2ExtrasXML")
                if isinstance(v2extras, dict):
                    title = v2extras.get("Title")
                else:
                    # Sometimes the source may already be flattened
                    title = d.get("V2ExtrasXML.Title")

                url = d.get("V2DocId") or d.get("DocumentIdentifier") or d.get("url")
                date = d.get("V21Date") or d.get("date")

                parts = [p for p in [date, title, url] if p]
                if parts:
                    lines.append("- " + " | ".join(str(p) for p in parts))

            if total is not None:
                lines.append(f"(Matched about {total} articles in total.)")
            return "\n".join(lines)

        return "No matching articles were found for that question. Try broadening the time range or using a different keyword."
