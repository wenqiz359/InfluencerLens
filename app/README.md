# InfluencerLens App

This is the backend container for the Influencer Discovery and Matching System.
It integrates **Neo4j (graph database)** with a **Python retrieval pipeline** for influencer search and matching.
There is **no frontend or API server yet** — all interactions happen directly via the terminal.

---

## Quick Start

### Build the Containers

In the project root (where `docker-compose.yml` is located), run:

```bash
docker compose build
```

This builds two services:

* **neo4j-db** → Neo4j 5.22 with APOC and Graph Data Science plugins
* **influencerlens-app** → Python backend container (retrieval + ranking pipeline)

If you changed `requirements.txt` or `Dockerfile`, rebuild cleanly:

```bash
docker compose build --no-cache
```

---

### Run the Pipeline Interactively

Since there’s no frontend or API yet, you can run the retrieval pipeline interactively:

```bash
docker compose run --rm --service-ports app bash -lc "python main.py"
```

You can input queries or commands directly in the console once it starts.

---

### Run Full Stack or View Logs

Run both the Neo4j and Python pipeline together:

```bash
docker compose up
```

View logs:

```bash
docker compose logs -f app
docker compose logs -f neo4j-db
```

---


## Data Persistence

Neo4j data and indexes are stored in:

```
./neo4j_data_enqi/
```

This directory is **volume-mounted**, so your data persists after container restarts.

Check existing indexes:

```bash
docker compose exec neo4j-db cypher-shell -u neo4j -p password -d neo4j \
"SHOW INDEXES YIELD name, type, entityType, state, labelsOrTypes, properties ORDER BY name"
```

If vector or fulltext indexes are missing, initialize them:

```bash
docker compose exec neo4j-db cypher-shell -u neo4j -p password -d neo4j \
-f /app/docker/init_neo4j.cypher
```
