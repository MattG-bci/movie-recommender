test:
    pytest -n 4 tests/

lint:
    poetry run ruff format src/ tests/

deploy-db:
    cd sqitch && sqitch deploy -d movie_recommender

revert-db:
    cd sqitch && sqitch revert -d movie_recommender

verify-db:
    cd sqitch && sqitch verify -d movie_recommender

pipeline *ARGS:
    poetry run python3 src/main.py {{ARGS}}

api:
    poetry run uvicorn api.app:app --reload

mobile-ios:
    #!/usr/bin/env bash
    set -euo pipefail
    open -a Simulator
    xcrun simctl boot "iPhone 16 Pro" >/dev/null 2>&1 || true
    poetry run uvicorn api.app:app --reload --port 8000 &
    api_pid=$!
    trap 'kill "$api_pid" 2>/dev/null || true' EXIT INT TERM
    cd mobile
    flutter run --dart-define=API_BASE_URL=http://localhost:8000

test-mobile:
    cd mobile && flutter test
