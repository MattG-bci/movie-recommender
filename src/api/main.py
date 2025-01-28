from src.ingestion.infer_db_pd import load_db
from src.model.recommender import Recommender
from utils.model_size import compute_model_size


model = Recommender(100, 100)
print(compute_model_size(model))
df = load_db()
