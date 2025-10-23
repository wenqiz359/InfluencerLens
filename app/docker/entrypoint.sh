#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8000}"

NEO4J_URI="${NEO4J_URI:-bolt://neo4j-db:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

NEO4J_INIT="${NEO4J_INIT:-1}"         # 1=初始化，0=跳过（默认跳过）
INIT_MARK_FILE="/app/.neo4j_inited"

VEC_DIM="${VEC_DIM:-1024}"
VEC_SIM="${VEC_SIM:-cosine}"

# ---------- Helpers ----------
wait_for_neo4j() {
  echo "Waiting for Neo4j at ${NEO4J_URI} ..."
  python - <<'PY'
import os, time, sys
from neo4j import GraphDatabase
uri=os.getenv("NEO4J_URI","bolt://neo4j:7687")
user=os.getenv("NEO4J_USER","neo4j")
pwd=os.getenv("NEO4J_PASSWORD","password")
for i in range(60):
    try:
        with GraphDatabase.driver(uri, auth=(user,pwd)) as drv, drv.session() as s:
            s.run("RETURN 1").consume()
            print("Neo4j is ready.")
            sys.exit(0)
    except Exception as e:
        time.sleep(2)
print("Neo4j did not become ready in time.", file=sys.stderr)
sys.exit(1)
PY
}

run_init() {
  echo "Initializing Neo4j schema (constraints / full-text / vector indexes) via Python ..."
  python - <<'PY'
import os, re
from neo4j import GraphDatabase

uri=os.getenv("NEO4J_URI","bolt://neo4j:7687")
user=os.getenv("NEO4J_USER","neo4j")
pwd=os.getenv("NEO4J_PASSWORD","password")
vec_dim=int(os.getenv("VEC_DIM","1024"))
vec_sim=os.getenv("VEC_SIM","cosine")

# 读取并替换参数占位（支持 $VEC_DIM / $VEC_SIM 两种写法）
path="docker/init_neo4j.cypher"
with open(path, "r", encoding="utf-8") as f:
    cypher=f.read()

# 用参数变量（不要在 cypher 文件里加引号）
cypher = cypher.replace("$VEC_DIM", str(vec_dim))
cypher = cypher.replace("$VEC_SIM", vec_sim)

# 简单按分号切句（跳过空行/注释）
stmts=[s.strip() for s in cypher.split(";") if s.strip() and not s.strip().startswith("//")]

driver=GraphDatabase.driver(uri, auth=(user,pwd))
with driver.session() as sess:
    for st in stmts:
        sess.run(st).consume()
print("Init done.")
PY
  touch "${INIT_MARK_FILE}"
  echo "Neo4j initialization completed."
}

start_app() {
  if [ -f "app.py" ]; then
    echo "Starting API server (uvicorn) on ${APP_HOST}:${APP_PORT} ..."
    python -m uvicorn app:app --host "${APP_HOST}" --port "${APP_PORT}" --workers 2
  else
    echo "Starting script: main.py ..."
    python main.py
  fi
}

# ---------- Main ----------
wait_for_neo4j

if [ "${NEO4J_INIT}" = "1" ] && [ ! -f "${INIT_MARK_FILE}" ]; then
  run_init
else
  echo "Skip Neo4j init (NEO4J_INIT=${NEO4J_INIT}, marker exists: $( [ -f "${INIT_MARK_FILE}" ] && echo yes || echo no ))"
fi

start_app
