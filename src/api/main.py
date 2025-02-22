from src.model.recommender import Recommender
from src.utils.model_size import compute_model_size


model = Recommender(100, 100)
print(compute_model_size(model))
