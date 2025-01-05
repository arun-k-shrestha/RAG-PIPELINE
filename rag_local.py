import os, json, pathlib, re, yaml
from dataclasses import dataclass
from typing import List, Dict, Any
from rich import print as rprint
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

# ---------------- Tokenizer ----------------
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def tok(s: str) -> List[int]:
    return enc.encode(s)

def detok(ids: List[int]) -> str:
    return enc.decode(ids)

SPEAKER_RE = re.compile(r"^\s*([A-Z][A-Za-z .&'-]{1,40}):\s*", re.M)  # e.g., "Operator:", "Tim Cook:", "CFO:"
print(SPEAKER_RE.findall("Operator: Hello\nTim Cook: Hi\nCFO: Bye"))