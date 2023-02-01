
for corpus in arguana  bioasq  climate-fever  dbpedia-entity  fever  fiqa  hotpotqa  nfcorpus  nq  quora  robust04  scidocs  scifact  signal1m   trec-news  webis-touche2020
do  
    # wget https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$corpus.zip -P /store2/scratch/oogundep/beir
    # unzip /store2/scratch/oogundep/beir/$corpus.zip -d /store2/scratch/oogundep/beir


    CUDA_VISIBLE_DEVICES=7 python -m pyserini.encode input   --corpus /store2/scratch/n3thakur/beir-datasets/$corpus/corpus.jsonl \
        --fields title text output  --embeddings /store2/scratch/oogundep/beir-index/$corpus \
        --to-faiss encoder --encoder /home/oogundep/odunayo/gtr-base-pth \
        --encoder-class gtr  --fields title text --dimension 768 \
        --batch 64 --max-length 512

    echo "======================================================"
    echo $corpus
    python -m pyserini.search.faiss --index /store2/scratch/oogundep/beir-index/$corpus  \
        --topics beir-v1.0.0-$corpus-test --output runs/run.beir-flat.$corpus.gtr-base.txt \
        --batch 128 --threads 12  --encoder /home/oogundep/odunayo/gtr-base-pth --tokenizer t5-base  \
        --encoder-class gtr
    
    echo "Results:"
    python -m pyserini.eval.trec_eval   -c -m ndcg_cut.10 beir-v1.0.0-$corpus-test runs/run.beir-flat.$corpus.gtr-base.txt
    echo "======================================================"
done