python -m pyserini.dsearch \
    --topics dl19-passage \
    --index /home/oogundep/pyserini/indexes/msmarco_new_faiss \
    --encoder /home/oogundep/odunayo/sentencepiece.model \
    --encoder-class st5 \
    --batch-size 128 \
    --threads 12 \
    --output runs/run.msmarco-passage.dl-19.st5.txt