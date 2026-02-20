# backend/services/context_manager.py

class ContextManager:
    """
    Shapes and truncates results to fit the LLM's context window.
    """

    def shape_results(self, es_response: dict, query_type: str) -> dict:
        """
        Takes raw Elasticsearch response and returns a compact, LLM-friendly representation.
        """
        # Placeholder: Implement result shaping logic
        return es_response

    def estimate_tokens(self, text: str) -> int:
        """
        Rough token count estimate.
        """
        return len(text.split())