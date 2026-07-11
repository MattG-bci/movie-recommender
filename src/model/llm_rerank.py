from anthropic import Anthropic

from etl.sql_queries import fetch_user_profile
from settings import LLMSettings
from schemas.recommendation import RecommendationOut, UserProfile


def rerank(
    user_id: int, prompt: str, exploration: float, k: int = 10
) -> list[RecommendationOut]:
    user_profile = fetch_user_profile(user_id, top_k=k)
    updated_prompt = update_prompt(prompt, user_profile, exploration)
    raw_llm_output = call_llm(updated_prompt)
    return raw_llm_output


def update_prompt(prompt: str, user_profile: UserProfile, exploration: float) -> str:
    enhancement = f"""

    This part is to enhance the original prompt message requesting a movie recommendation.

    The user's movie taste is represented by the following user profile:

    {user_profile.model_dump()}

    Consider this as a general idea of what user likes the most.

    For the reranking done by LLM, please consider the value of the exploration parameter

    Exploration: {exploration}

    where value 1.0 represents an eagerness to try out new movies, directors, actors etc. and value 0.0 represents
    a mood where recommendations should follow user profile. Anything in between is a spectrum, a balance between exploration of
    novelty and sticking to the original taste.

    """
    return prompt + enhancement


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
