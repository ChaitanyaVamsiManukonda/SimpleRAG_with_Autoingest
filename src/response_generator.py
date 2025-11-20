# src/response_generator.py
import requests
import os
from typing import Dict, Optional


class ResponseGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        max_tokens: int = 1000,
        temperature: float = 0.2,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        self.api_url = "https://api.anthropic.com/v1/messages"
        if not self.api_key:
            raise ValueError("Anthropic API key is required")

    def _create_system_prompt(self, context: str) -> str:
        """
        Create a system prompt with the retrieved context.

        The context is already formatted and labeled by the QueryProcessor.
        """
        return f"""You are a world-class AI assistant that answers user questions strictly based on the provided context.

- Use only the information in the CONTEXT to answer.
- If the context does not contain enough information, clearly say you do not have enough information rather than guessing.
- When the answer is partially supported, clearly separate supported facts from assumptions.
- Write a clear, concise answer in well-structured markdown.
- Prefer quoting or paraphrasing the most relevant chunks, and refer to their source labels when helpful.

CONTEXT:
{context}
"""

    def generate_response(self, query: str, context: str) -> Dict:
        """Generate a response using Claude API."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        system_prompt = self._create_system_prompt(context)

        payload = {
            "model": self.model,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Error calling Claude API: {response.text}")

        result = response.json()
        text = result["content"][0]["text"]

        return {
            "query": query,
            "response": text,
            "model": self.model,
        }
