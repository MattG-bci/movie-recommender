from etl import Recommender
from etl import compute_model_size


model = Recommender(100, 100)
print(compute_model_size(model))
