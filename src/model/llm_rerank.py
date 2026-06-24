from anthropic import Anthropic

from settings import LLMSettings
from schemas.recommendation import RecommendationOut


def rerank(prompt: str, exploration: float, k: int = 10) -> list[RecommendationOut]:
    raw_llm_output = call_llm(prompt)
    return raw_llm_output


def call_llm(prompt: str) -> str:
    settings = LLMSettings()
    model = Anthropic(api_key=settings.API_KEY)
    out = model.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=settings.MODEL,
    )
    return out.content[0].text
