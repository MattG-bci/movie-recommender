import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse



app = FastAPI()


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
    return JSONResponse(content=None)


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=6969)
