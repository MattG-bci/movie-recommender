import dspy

from etl.sql_queries import fetch_user_profile
from settings import LLMSettings
from schemas.recommendation import RecommendationOut, UserProfile, UserProfileWithID


async def rerank(
    user_id: int, prompt: str, exploration: float, k: int = 10
) -> list[RecommendationOut]:
    user_profile = await fetch_user_profile(user_id, top_k=k)
    updated_prompt = build_prompt(prompt, user_profile, exploration)
    raw_llm_output = ""  # call_llm(updated_prompt)
    return raw_llm_output + updated_prompt


def build_prompt(prompt: str, user_profile: UserProfile, exploration: float) -> str:
    enhanced_prompt = """

    This part is to enhance the original prompt message requesting a movie recommendation.

    The user's movie taste is represented by the following user profile:

    {user_taste_profile}

    Consider this as a general idea of what user likes the most.

    For the reranking done by LLM, please consider the value of the exploration parameter

    Exploration: {exploration}

    where value 1.0 represents an eagerness to try out new movies, directors, actors etc. and value 0.0 represents
    a mood where recommendations should follow user profile. Anything in between is a spectrum, a balance between exploration of
    novelty and sticking to the original taste.

    """
    user_taste_fields = {
        key: value
        for key, value in user_profile.model_dump().items()
        if key in UserProfileWithID.model_fields
    }
    enhanced_prompt = prompt + enhanced_prompt.format(
        user_taste_profile=user_taste_fields, exploration=exploration
    )
    return enhanced_prompt


def configure_llm() -> None:
    settings = LLMSettings()
    lm = dspy.LM(
        f"anthropic/{settings.MODEL}", api_key=settings.API_KEY, max_tokens=1024
    )
    dspy.configure(lm=lm)
