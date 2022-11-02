python -m pyserini.encode input   --corpus /home/oogundep/msmarco_data/collection_jsonl --fields text --shard-id 0 --shard-num 1 output  --embeddings /home/oogundep/pyserini/indexes/msmarco_new_9900 encoder --encoder st5 --fields text --batch 256 --fp16

#python -m pyserini.encode input   --corpus tests/resources/simple_cacm_corpus.json --fields text --shard-id 0 --shard-num 1 --delimiter "\n" output  --embeddings /home/oogundep/pyserini/indexes/test_corpus_faiss --to-faiss  encoder --encoder st5 --fields text --batch 8 --fp16



# python -m pyserini.index.faiss \
#   --input  /home/oogundep/pyserini/indexes/msmarco_new \
#   --output /home/oogundep/pyserini/indexes/msmarco_new_faiss \
#   --threads 5