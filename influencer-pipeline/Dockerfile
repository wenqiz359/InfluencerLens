FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocess.py .
COPY .env .

CMD ["python", "-u", "preprocess.py"]

