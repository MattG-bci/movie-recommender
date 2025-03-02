FROM python:3.12-slim
COPY . /app
WORKDIR /app


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -


ENV PATH="/root/.local/bin:$PATH"

RUN poetry --version

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false

RUN poetry install --no-interaction --no-ansi

CMD ["poetry", "run", "pytest"]
