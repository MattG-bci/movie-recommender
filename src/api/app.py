import datetime

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse


app = FastAPI()


@app.get("/")
async def get_status() -> dict[str, str]:
    return {"status": "OK"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)
