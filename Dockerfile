FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
