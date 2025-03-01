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
