lang=finnish
export CUDA_VISIBLE_DEVICES=6

python -m pyserini.search.faiss \
    --topics mrtydi-v1.1-$lang-test \
    --index /home/oogundep/pyserini/indexes/$lang-base \
    --encoder /home/oogundep/odunayo/mgtr/sentencepiece.model \
    --encoder-class st5 \
    --batch-size 256 \
    --threads 12 \
    --output runs/run.mrtydi-$lang.100k_checkpoint.mgtr-base.txt

python -m pyserini.eval.trec_eval -c -mrecip_rank -mrecall.100 mrtydi-v1.1-$lang-test runs/run.mrtydi-$lang.100k_checkpoint.mgtr-base.txt
