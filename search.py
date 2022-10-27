from pyserini.search.faiss import FaissSearcher, SentenceT5QueryEncoder
encoder = SentenceT5QueryEncoder()
searcher = FaissSearcher("/home/oogundep/pyserini/indexes/test_corpus_faiss", encoder)
hits = searcher.search("Random Correlated Normal Variables")
for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:7} {hits[i].score:.5f}')