FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 基础系统依赖
RUN apt-get update -o Acquire::Retries=5 \
 && apt-get install -y --no-install-recommends \
      curl ca-certificates netcat-traditional \
 && rm -rf /var/lib/apt/lists/*

# 先装依赖再拷代码，利用缓存
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝代码
COPY . /app

# 入口脚本权限
RUN chmod +x docker/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["bash", "docker/entrypoint.sh"]