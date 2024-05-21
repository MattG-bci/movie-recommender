import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
async def root():
    return "Hello World"

@app.get("/data")
async def hello():
    return JSONResponse(content={
        "date": datetime.datetime.now().isoformat(),
        "name": "Jeff",
        "occupation": "Student"
    })

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=6969)
