import mysql.connector
import os
import json

mydb = mysql.connector.connect(
  host="localhost",
  user=os.environ["USER"],
  password=os.environ["PASSWORD"],
  database="mysql"
)

cursor = mydb.cursor()

# Define the array
my_array = ['apple', 'banana', 'cherry']

# Create the table with three columns
cursor.execute("CREATE TABLE IF NOT EXISTS my_table (id INT AUTO_INCREMENT PRIMARY KEY, f1 VARCHAR(256), f2 VARCHAR(256), f3 VARCHAR(256))")

# Insert the array into the database
cursor.execute("INSERT INTO my_table (f1, f2, f3) VALUES (%s, %s, %s)", my_array)

# Commit the changes
mydb.commit()

# Close the cursor and connection
cursor.close()
mydb.close()
