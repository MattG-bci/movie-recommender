import mysql.connector
import os

mydb = mysql.connector.connect(
  host="localhost",
  user=os.environ["USER"],
  password=os.environ["PASSWORD"],
  database="mysql"
)
