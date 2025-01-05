import pathlib, yaml


# ---------------- Config ----------------
CFG = yaml.safe_load(open("rag_config.yaml"))
CHUNK_TOKENS = CFG["chunk_tokens"]
CHUNK_OVERLAP = CFG["chunk_overlap"]
TOP_K = CFG["retrieve_top_k"]
RERANK_TOP_K = CFG["rerank_top_k"]
INDEX_PATH = pathlib.Path(CFG["index_path"])
META_PATH = pathlib.Path(CFG["metadata_path"])
EMB_NAME = CFG["embedding_model"]
RERANK_NAME = CFG["reranker_model"]
SPEAKER_CHUNKING = CFG.get("speaker_chunking", True)
