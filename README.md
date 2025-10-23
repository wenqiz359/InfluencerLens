# InfluencerLens System

InfluencerLens is a modular backend for influencer data processing, graph storage (Neo4j), and semantic retrieval/reranking.

## Prerequisites
- Docker Desktop
- (Optional) OpenAI API key if you want "Why matched" explanations

## Project Structure
```
milestone2/
├── app/                        # Retrieval + reranking pipeline
│   ├── main.py
│   ├── retrieve_dual.py
│   ├── parse_or_rewrite.py
│   ├── fusion_rerank_pipeline.py
│   └── docker/entrypoint.sh
├── influencer-pipeline/
│   └── shared/influencer_index.jsonl
├── neo4j_data_enqi/            # Neo4j persisted data (created at runtime)
├── outputs/                    # Logs/exports (created at runtime)
├── docker-compose.enqi.yml     # Neo4j + uploader + app
├── Dockerfile                  # Uploader image
├── app/Dockerfile              # App image
├── load_influencer_hashtags.py # Data uploader (creates indexes + loads data)
├── requirements.txt
└── .env                        # Runtime configuration (you create this)
```

## .env (create in repo root)
```
NEO4J_URI=bolt://neo4j-db:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

NEO4J_INIT=0
NEO4J_LABEL=Influencer
NEO4J_FT_INDEX=influencer_ft
NEO4J_VEC_INDEX=influencer_summary_vec
VEC_DIM=1024
VEC_SIM=cosine

APP_HOST=0.0.0.0
APP_PORT=8000

# Optional (for "Why matched" explanations)
OPENAI_API_KEY=sk-...
```

## Quick Start (Single Command)
Runs database + uploader + interactive app:
```bash
docker compose -f docker-compose.enqi.yml up neo4j-db enqi-uploader --build -d && \
sleep 60 && \
docker compose -f docker-compose.enqi.yml run --rm --service-ports app bash -lc "python main.py"
```

- Starts Neo4j with plugins
- Uploader creates constraints, fulltext and vector indexes, and ingests data
- App runs interactively and connects to Neo4j

## Clean and Re-run
```bash
docker compose -f docker-compose.enqi.yml down
docker system prune -a -f --volumes
rm -rf neo4j_data_enqi/ outputs/ app/outputs/
docker compose -f docker-compose.enqi.yml up neo4j-db enqi-uploader --build -d && \
sleep 60 && \
docker compose -f docker-compose.enqi.yml run --rm --service-ports app bash -lc "python main.py"
```

## Running Pieces Individually
- Start only Neo4j:
```bash
docker compose -f docker-compose.enqi.yml up neo4j-db --build -d
```
- Load data only:
```bash
docker compose -f docker-compose.enqi.yml up enqi-uploader --build
```
- Run app (interactive):
```bash
docker compose -f docker-compose.enqi.yml run --rm --service-ports app bash -lc "python main.py"
```

## Troubleshooting
- EOFError: App expects input; run interactively:
```bash
docker compose -f docker-compose.enqi.yml run --rm --service-ports app bash -lc "python main.py"
```
- Index missing (e.g., influencer_summary_vec): ensure uploader runs after Neo4j is ready (use the single command above). The uploader creates both fulltext and vector indexes automatically.
- Neo4j not ready in time: increase wait (e.g., `sleep 90`) or check logs:
```bash
docker compose -f docker-compose.enqi.yml logs neo4j-db
```
- OpenAI error in "Why matched": set `OPENAI_API_KEY` in `.env`.

## Access
- Neo4j Browser: http://localhost:7474 (neo4j/password)
- App: runs as an interactive console (prompts for queries)
