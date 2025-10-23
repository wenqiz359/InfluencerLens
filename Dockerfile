# =======================================
# Enqi Liu - Neo4j Uploader Container
# =======================================
FROM python:3.10-slim

WORKDIR /app

# ✅ 修正版：安装 netcat-openbsd 替代 netcat
RUN apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "load_influencer_hashtags.py"]