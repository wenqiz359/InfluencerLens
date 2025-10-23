from neo4j import GraphDatabase, exceptions
import json
from tqdm import tqdm
import os

# ==========================
# 1. CONNECT TO NEO4J
# ==========================
def connect_to_neo4j(
    uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    user=os.getenv("NEO4J_USER", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "password123")  
):
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database="neo4j") as session:
            result = session.run("RETURN 'Connected!' AS msg")
            print("✅", result.single()["msg"])
        return driver
    except exceptions.AuthError as e:
        print("❌ Authentication failed. Please verify your Neo4j username/password.")
        print(f"   Current -> uri={uri}, user={user}, password=******")
        raise
    except Exception as e:
        print("❌ Could not connect to Neo4j.")
        print("Error:", e)
        print("💡 Tip: Make sure your Neo4j Docker container is running (`docker ps`).")
        raise


# ==========================
# 2. SETUP CONSTRAINTS
# ==========================
def setup_constraints(session):
    print("🧱 Setting up database constraints and indexes...")
    session.run("""
        CREATE CONSTRAINT influencer_name_unique IF NOT EXISTS
        FOR (i:Influencer)
        REQUIRE i.name IS UNIQUE
    """)
    session.run("""
        CREATE CONSTRAINT hashtag_name_unique IF NOT EXISTS
        FOR (h:Hashtag)
        REQUIRE h.name IS UNIQUE
    """)
    # Fulltext index for BM25-style search
    session.run("""
        CREATE FULLTEXT INDEX influencer_ft IF NOT EXISTS
        FOR (i:Influencer)
        ON EACH [i.summary, i.combined_text]
    """)
    session.run("""
        CREATE VECTOR INDEX influencer_summary_vec IF NOT EXISTS
        FOR (i:Influencer) ON (i.embedding_summary)
        OPTIONS { indexConfig: {
          `vector.dimensions`: 1024,
          `vector.similarity_function`: 'cosine'
        }}
    """)
    print("✅ Constraints + Fulltext index + Vector index ready.")


# ==========================
# 3. INSERT ONE INFLUENCER
# ==========================
def insert_influencer(tx, data):
    name = data["name"]
    summary = data.get("summary", "")
    emb_raw = data.get("embedding_raw", [])
    emb_summary = data.get("embedding_summary", [])
    meta = data.get("metadata", {})

    tx.run("""
        MERGE (i:Influencer {name: $name})
        SET i.summary = $summary,
            i.embedding_raw = $emb_raw,
            i.embedding_summary = $emb_summary,
            i.followers = $followers,
            i.signature = $signature,
            i.likes_avg = $likes_avg,
            i.comments_avg = $comments_avg,
            i.shares_avg = $shares_avg,
            i.plays_avg = $plays_avg,
            i.collects_avg = $collects_avg,
            i.avg_video_duration_sec = $avg_video_duration_sec,
            i.created_at = $created_at,
            i.source_url = $source_url,
            i.combined_text = $combined_text
    """, name=name, summary=summary,
         emb_raw=emb_raw, emb_summary=emb_summary,
         followers=meta.get("followers"),
         signature=meta.get("signature"),
         likes_avg=meta.get("likes_avg"),
         comments_avg=meta.get("comments_avg"),
         shares_avg=meta.get("shares_avg"),
         plays_avg=meta.get("plays_avg"),
         collects_avg=meta.get("collects_avg"),
         avg_video_duration_sec=meta.get("avg_video_duration_sec"),
         created_at=meta.get("created_at"),
         source_url=meta.get("source_url"),
         combined_text=meta.get("combined_text"))

    # Insert hashtags and relationships
    for tag in meta.get("hashtags", []):
        if not tag:
            continue
        tx.run("""
            MERGE (h:Hashtag {name: $tag})
            WITH h
            MATCH (i:Influencer {name: $name})
            MERGE (i)-[:USES_HASHTAG]->(h)
        """, tag=tag.strip().lower(), name=name)


# ==========================
# 4. LOAD JSONL FILE
# ==========================
def create_indexes(driver):
    """Create required indexes if they don't exist"""
    with driver.session() as session:
        try:
            # Create constraints
            session.run("CREATE CONSTRAINT infl_name IF NOT EXISTS FOR (i:Influencer) REQUIRE i.name IS UNIQUE")
            session.run("CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (h:Hashtag) REQUIRE h.name IS UNIQUE")
            
            # Create fulltext index
            session.run("""
                CREATE FULLTEXT INDEX influencer_ft IF NOT EXISTS
                FOR (i:Influencer) ON EACH [i.combined_text, i.summary]
                OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard' } }
            """)
            
            # Create vector index
            session.run("""
                CREATE VECTOR INDEX influencer_summary_vec IF NOT EXISTS
                FOR (i:Influencer) ON (i.embedding_summary)
                OPTIONS { indexConfig: {
                  `vector.dimensions`: 1024,
                  `vector.similarity_function`: 'cosine'
                }}
            """)
            
            print("✅ Indexes created successfully")
        except Exception as e:
            print(f"⚠️ Index creation warning: {e}")
def load_data(filepath=None):
    # ✅ Smart file resolution: auto-detect correct path
    candidate_paths = [
        filepath,
        os.getenv("DATA_FILE"),
        "./influencer_index.jsonl",
        "/shared/influencer_index.jsonl"
    ]
    filepath = next((p for p in candidate_paths if p and os.path.exists(p)), None)

    if not filepath:
        raise FileNotFoundError("❌ Could not find influencer_index.jsonl in ./ or /shared/. Check volume mount.")

    print(f"📂 Using data file: {filepath}")
    driver = connect_to_neo4j()
    create_indexes(driver)

    with driver.session() as session:
        setup_constraints(session)

        # Load JSONL
        with open(filepath, "r") as f:
            data = [json.loads(line) for line in f]
        print(f"📦 Loaded {len(data)} influencers from {filepath}")

        # Insert all influencers
        with session.begin_transaction() as tx:
            for record in tqdm(data, desc="Loading influencers"):
                insert_influencer(tx, record)
            tx.commit()

    print("✅ All influencer and hashtag data loaded successfully.")
    driver.close()


# ==========================
# 5. ENTRY POINT
# ==========================
if __name__ == "__main__":
    load_data()
