#python -m pyserini.encode input   --corpus /home/oogundep/msmarco_data/collection_jsonl --fields text --shard-id 0 --shard-num 1 output  --embeddings /home/oogundep/pyserini/indexes/msmarco_new_9900 encoder --encoder st5 --fields text --batch 256 --fp16
lang=$1
cuda=$2

export CUDA_VISIBLE_DEVICES=${cuda}
# export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"

python3 -m pyserini.encode input --corpus /home/oogundep/odunayo/mrtydi/mrtydi-v1.1-${lang}/collection/docs.jsonl --fields title text --shard-id 0 --shard-num 1 --delimiter "\n\n" output --embeddings indexes/512/${lang}-base --to-faiss encoder --encoder /home/oogundep/odunayo/mgtr-512/checkpoint_1100000 --encoder-class st5 --fields title text --batch 64 --fp16 --device cuda:${cuda}

# python -m pyserini.index.faiss \
#   --input  /home/oogundep/pyserini/indexes/msmarco_new \
#   --output /home/oogundep/pyserini/indexes/msmarco_new_faiss \
#   --threads 5

##updated model for korean
