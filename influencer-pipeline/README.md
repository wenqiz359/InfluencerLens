# Influencer Pipeline — Data Preprocessing

Run data preprocessing to generate `/shared/influencer_index.jsonl`:

```bash
cd ~/Desktop/influencer-pipeline
docker compose -f docker-compose-preprocess.yml build --no-cache
docker compose -f docker-compose-preprocess.yml up
```
