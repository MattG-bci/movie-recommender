import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.ingestion.infer_db_pd import load_db
from json import loads


app = FastAPI()
df = load_db()


@app.get("/")
async def root():
    return JSONResponse(
        content={
            "date": datetime.datetime.now().isoformat(),
            "name": "Jeff",
            "occupation": "Student",
        }
    )


@app.get("/data")
async def display_data():
    global df
    data_point = df.iloc[0, [1, 2]]
    result = loads(data_point.to_json())
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=6969)
