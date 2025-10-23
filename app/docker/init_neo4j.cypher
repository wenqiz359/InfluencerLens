// 1) 唯一约束
CREATE CONSTRAINT infl_name IF NOT EXISTS
FOR (i:Influencer) REQUIRE i.name IS UNIQUE;

CREATE CONSTRAINT tag_name IF NOT EXISTS
FOR (h:Hashtag) REQUIRE h.name IS UNIQUE;

// 2) Full-Text (BM25/Lucene)
CREATE FULLTEXT INDEX influencer_ft IF NOT EXISTS
FOR (i:Influencer) ON EACH [i.combined_text, i.summary]
OPTIONS { indexConfig: { `fulltext.analyzer`: 'standard' } };

// 3) Vector index: embedding_summary
CREATE VECTOR INDEX influencer_summary_vec IF NOT EXISTS
FOR (i:Influencer) ON (i.embedding_summary)
OPTIONS { indexConfig: {
  `vector.dimensions`: 1024,
  `vector.similarity_function`: 'cosine'
}};

