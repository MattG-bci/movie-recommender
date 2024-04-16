from flask import Flask
import datetime

app = Flask(__name__)

@app.route("/data")
def hello():
    return {
        "date": datetime.datetime.now(),
        "name": "Jeff",
        "occupation": "Student"
    }

if __name__ == "__main__":
    app.run("localhost", 6969)
