export CUDA_VISIBLE_DEVICES=6
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"

#python -m pyserini.encode input   --corpus /home/oogundep/msmarco_data/collection_jsonl/shard3 --fields text --shard-id 0 --shard-num 1 output  --embeddings /home/oogundep/pyserini/indexes/msmarco_20k/shard3 encoder --encoder /home/oogundep/odunayo/20221101/checkpoint_1020000 --encoder-class st5 --fields text --batch 256 --fp16

#python -m pyserini.encode input   --corpus tests/resources/simple_cacm_corpus.json --fields text --shard-id 0 --shard-num 1 --delimiter "\n" output  --embeddings /home/oogundep/pyserini/indexes/test_corpus_faiss --to-faiss  encoder --encoder st5 --fields text --batch 8 --fp16



python -m pyserini.index.faiss \
  --input  /home/oogundep/pyserini/indexes/msmarco_100k \
  --output /home/oogundep/pyserini/indexes/msmarco_100k_faiss \
  --threads 8