import mysql.connector
import os


mydb = mysql.connector.connect(
    host="localhost",
    user=os.environ["USER"],
    password=os.environ["PASSWORD"],
    database="mysql",
)

cursor = mydb.cursor()
create_users_table_query = """
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL
)
"""
cursor.execute(create_users_table_query)

# Create the ratings table
create_ratings_table_query = """
CREATE TABLE IF NOT EXISTS ratings (
    rating_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    movie_name VARCHAR(100) NOT NULL,
    rating INT NOT NULL,
    CONSTRAINT fk_ratings_users FOREIGN KEY (user_id) REFERENCES users(user_id)
)
"""
cursor.execute(create_ratings_table_query)

user_name = "no_real_username"
movie_name = "Avatar"
rating = 9.0

insert_user_query = "INSERT INTO users (username) VALUES (%s)"
cursor.execute(insert_user_query, (user_name,))

cursor.execute("SELECT LAST_INSERT_ID()")
user_id = cursor.fetchone()[0]

insert_rating_query = (
    "INSERT INTO ratings (user_id, movie_name, rating) VALUES (%s, %s, %s)"
)
cursor.execute(insert_rating_query, (user_id, movie_name, rating))

mydb.commit()
cursor.close()
mydb.close()
