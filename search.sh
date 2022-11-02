export CUDA_VISIBLE_DEVICES=0

python -m pyserini.dsearch \
    --topics msmarco-passage-dev-subset \
    --index /home/oogundep/pyserini/indexes/msmarco_official_faiss \
    --encoder /home/oogundep/odunayo/sentencepiece.model \
    --encoder-class st5 \
    --batch-size 256 \
    --threads 12 \
    --output runs/run.msmarco-passage.new.st5.txt