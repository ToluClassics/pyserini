#for corpus in fiqa
#do  
    #wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$corpus.zip -P /home/mac/beir
    #unzip /home/mac/beir/$corpus.zip -d /home/mac/beir


#    python3 index_gtr_models.py input   --corpus /home/mac/beir/$corpus/corpus.jsonl \
#    --fields title text output  --embeddings /home/mac/beir-index/$corpus \
#    --to-faiss encoder --encoder gs://t5-data/pretrained_models/t5x/retrieval/gtr_base  --encoder-class st5  --fields title text --dimension 768 \
#    --batch 128 --max-length 512
#done


for corpus in fiqa
do
    echo "======================================================"
    echo $corpus
    python3 -m pyserini.search.faiss --index /home/mac/beir-index/$corpus  \
        --topics beir-v1.0.0-$corpus-test --output runs/run.beir-flat.$corpus.gtr-base.txt \
        --batch 128 --threads 12  --encoder gs://t5-data/pretrained_models/t5x/retrieval/gtr_base \
        --encoder-class st5
    
    echo "Results:"
    python -m pyserini.eval.trec_eval   -c -m ndcg_cut.10 beir-v1.0.0-$corpus-test runs/run.beir-flat.$corpus.gtr-base.txt
    echo "======================================================"
done

