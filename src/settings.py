import os

from dotenv import load_dotenv

load_dotenv()


USERNAME_PAGE = os.getenv("USERNAME_PAGE")
RATINGS_PAGE = os.getenv("RATINGS_PAGE")
DB_HOST = os.getenv("HOST")
DB_USER = os.getenv("USER")
DB_PASS = os.getenv("PASS")
DB_NAME = os.getenv("NAME")
DB_PORT = os.getenv("PORT")
