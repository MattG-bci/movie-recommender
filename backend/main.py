from src.data_processing.web_scraping import UserScraper, RatingScraper
from src.data_processing.infer_db_pd import transform_data
from src.data_processing.infer_db_pd import load_db, append_data_samples, save_db
from utils.config import config
from src.model.dataloader import MovieDataloader
from src.model.recommender import Recommender
from utils.model_size import compute_model_size


model = Recommender(100, 100)
print(compute_model_size(model))
df = load_db()
