from .healthcheck import health as health_router  # noqa: F401
from .ratings import ratings as ratings_router  # noqa: F401
from .movies import movies as movies_router  # noqa: F401
from .users import users as users_router  # noqa: F401
from .recommendations import recommendations as recommendations_router  # noqa: F401


__all__ = ["healthcheck", "ratings", "movies", "users", "recommendations"]
