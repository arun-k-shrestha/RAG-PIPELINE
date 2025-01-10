import os, json, pathlib, re, yaml
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from rich import print as rprint
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
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


# This is a simple rule-based speaker segmentation.
def split_by_speaker(text: str) -> List[Dict[str, Any]]:
    """
    Returns segments like: {"speaker": "CFO", "text": "..."}.
    If no speaker labels, returns one segment with speaker=None.
    """
    positions = [(m.start(), m.group(1)) for m in SPEAKER_RE.finditer(text)]
    if not positions:
        return [{"speaker": None, "text": text}]
    segments = []
    for i, (pos, spk) in enumerate(positions):
        end = positions[i+1][0] if i+1 < len(positions) else len(text)
        seg_text = text[pos:end]
        # Strip leading "Speaker: "
        seg_text = re.sub(r"^\s*"+re.escape(spk)+r":\s*", "", seg_text, count=1, flags=re.M)
        segments.append({"speaker": spk.strip(), "text": seg_text.strip()})
    return segments

# This chunks a segment of text (with optional speaker) into smaller chunks with overlap.
def chunk_segment(seg_text: str, doc_id: str, speaker: Optional[str], chunk_tokens=700, overlap_tokens=120):
    ids = tok(seg_text)
    n = len(ids)
    out = []
    start = 0
    cid = 0
    while start < n:
        end = min(n, start + chunk_tokens)
        sub = ids[start:end]
        text = detok(sub)
        out.append({
            "chunk_id": f"{doc_id}#{speaker or 'UNK'}#{cid}",
            "doc_id": doc_id,
            "speaker": speaker,
            "text": text
        })
        cid += 1
        if end == n: break
        start = end - overlap_tokens
    return out

# This function calls to chunk text, with or without speaker segmentation.
def semantic_chunks(text: str, doc_id: str) -> List[Dict[str, Any]]:
    if SPEAKER_CHUNKING:
        segs = split_by_speaker(text)
        chunks = []
        for seg in segs:
            chunks.extend(chunk_segment(seg["text"], doc_id, seg["speaker"], CHUNK_TOKENS, CHUNK_OVERLAP))
        return chunks
    else:
        print("No speaker chunking")
        return chunk_segment(text, doc_id, None, CHUNK_TOKENS, CHUNK_OVERLAP)
    
# ---------------- Load docs ----------------
DATA_DIR = pathlib.Path("data")
docs = []
for p in sorted(DATA_DIR.glob("*.txt")):
    docs.append({"doc_id": p.stem, "text": p.read_text(encoding="utf-8")})
    print(p.resolve())
if not docs:
    raise SystemExit("No docs found in data/*.txt")


all_chunks: List[Dict[str,Any]] = []

for d in docs:
    all_chunks.extend(semantic_chunks(d["text"],d["doc_id"]))
    rprint(f"[bold green]Chunked[/] {len(all_chunks)} chunks from {len(docs)} docs.")

# --------------- Embeddings ---------------
EMB_NAME = "all-MiniLM-L6-v2"
emb_model = SentenceTransformer(EMB_NAME)

def embed_texts(texts:List[str]) -> np.ndarray:
    embeddings = emb_model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")

# ---------------- FAISS index ----------------

dimension = emb_model.get_sentence_embedding_dimension()

print(faiss.IndexFlatIP(dimension))

def build_or_load_index(chunks: List[Dict[str,Any]]):
    if INDEX_PATH.exists() or META_PATH.exists():
        rprint("[green]Loading existing FAISS index…[/]")
        idx = faiss.read_index(str(INDEX_PATH))
        meta = [json.loads(l) for l in open(META_PATH, "r", encoding="utf-8")]
        return idx, meta
    texts = [c["text"] for c in chunks]
    vecs = embed_texts(texts)
    idx = faiss.IndexFlatIP(dimension)
    idx.add(vecs)
    faiss.write_index(idx,str(INDEX_PATH))
    with open(META_PATH, "w",encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return idx, chunks
       
index, meta = build_or_load_index(all_chunks)