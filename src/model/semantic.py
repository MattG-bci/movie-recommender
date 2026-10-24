from collections.abc import Sequence
from math import sqrt

import dspy

from schemas.movie import Movie
from settings import EmbeddingSettings


def build_movie_profile(movie: Movie) -> str:
    """Build the stored semantic representation from known movie metadata."""
    parts = [movie.title, str(movie.release_year)]
    if movie.description:
        parts.append(movie.description)
    if movie.genres:
        parts.append(f"Genres: {', '.join(movie.genres)}")
    if movie.director:
        parts.append(f"Director: {movie.director}")
    if movie.actors:
        parts.append(f"Cast: {', '.join(movie.actors)}")
    if movie.country:
        parts.append(f"Country: {movie.country}")
    return ". ".join(parts)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")

    denominator = sqrt(sum(value * value for value in left)) * sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / denominator


def normalize_similarity_scores(scores: dict[int, float]) -> dict[int, float]:
    """Map a request's similarities to 1--10 by percentile, preserving ties."""
    if not scores:
        return {}
    values = list(scores.values())
    if len(values) == 1 or min(values) == max(values):
        return {movie_id: 5.5 for movie_id in scores}

    ordered = sorted(scores.items(), key=lambda item: item[1])
    normalized: dict[int, float] = {}
    index = 0
    total = len(ordered)
    while index < total:
        end = index + 1
        while end < total and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        score = 1 + 9 * (average_rank - 1) / (total - 1)
        for movie_id, _ in ordered[index:end]:
            normalized[movie_id] = score
        index = end
    return normalized


def score_semantic_query(
    query_embedding: Sequence[float], movie_embeddings: dict[int, Sequence[float]]
) -> dict[int, float]:
    similarities = {
        movie_id: cosine_similarity(query_embedding, embedding)
        for movie_id, embedding in movie_embeddings.items()
    }
    return normalize_similarity_scores(similarities)


def blend_scores(
    cf_scores: dict[int, float],
    text_scores: dict[int, float] | None = None,
    image_scores: dict[int, float] | None = None,
    cf_weight: float = 0.55,
    text_weight: float = 0.25,
    image_weight: float = 0.20,
) -> dict[int, float]:
    active = [(cf_scores, cf_weight)]
    if text_scores is not None:
        active.append((text_scores, text_weight))
    if image_scores is not None:
        active.append((image_scores, image_weight))
    total_weight = sum(weight for _, weight in active)
    return {
        movie_id: sum(scores[movie_id] * weight for scores, weight in active)
        / total_weight
        for movie_id in cf_scores
    }


class EmbeddingClient:
    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self.settings = settings or EmbeddingSettings()
        self._embedder = dspy.Embedder(
            f"openai/{self.settings.MODEL}", api_key=self.settings.API_KEY
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self._embedder(texts)]
