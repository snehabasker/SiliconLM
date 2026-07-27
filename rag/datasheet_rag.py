"""Chunk .txt/.md docs, embed + index with FAISS, answer questions with citations.

Refuses to answer if nothing scores above MIN_SCORE, instead of guessing.

python datasheet_rag.py build --docs docs_dir/ --index index/
python datasheet_rag.py ask --index index/ --q "absolute max VDD?"
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

EMB_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_CHARS = 900
OVERLAP = 150
MIN_SCORE = 0.45  # cosine floor: below this the engine refuses to answer
TOP_K = 4


def chunk(text: str) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + CHUNK_CHARS])
        i += CHUNK_CHARS - OVERLAP
    return [c.strip() for c in out if len(c.strip()) > 80]


def build(docs_dir: Path, index_dir: Path) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMB_MODEL)
    passages, meta = [], []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        for j, c in enumerate(chunk(path.read_text(encoding="utf-8", errors="ignore"))):
            passages.append(c)
            meta.append({"source": path.name, "chunk": j})
    if not passages:
        raise SystemExit("no .txt/.md documents found")
    emb = model.encode(passages, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.asarray(emb, dtype=np.float32))
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "faiss.idx"))
    (index_dir / "meta.pkl").write_bytes(pickle.dumps({"passages": passages, "meta": meta}))
    print(f"indexed {len(passages)} chunks from {docs_dir}")


def retrieve(index_dir: Path, query: str) -> list[dict]:
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMB_MODEL)
    index = faiss.read_index(str(index_dir / "faiss.idx"))
    store = pickle.loads((index_dir / "meta.pkl").read_bytes())
    q = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.asarray(q, dtype=np.float32), TOP_K)
    hits = []
    for s, i in zip(scores[0], ids[0]):
        if i == -1 or s < MIN_SCORE:
            continue
        hits.append({"score": float(s), "text": store["passages"][i],
                     **store["meta"][i]})
    return hits


def answer(index_dir: Path, query: str, llm_fn=None) -> dict:
    """Return {"answer", "citations"} — refuses when retrieval is thin."""
    hits = retrieve(index_dir, query)
    if not hits:
        return {"answer": "Not found in the indexed documentation — refusing to "
                          "guess. Add the relevant datasheet and rebuild the index.",
                "citations": []}
    context = "\n\n".join(
        f"[{h['source']} p.{h['chunk']}] {h['text']}" for h in hits)
    if llm_fn is None:  # extractive fallback: quote the best passages
        text = "\n\n".join(
            f"{h['text'][:400]}\n[source: {h['source']} p.{h['chunk']}]" for h in hits[:2])
        return {"answer": text, "citations": hits}
    prompt = (
        "Answer strictly from the context. Cite every claim as "
        "[source: file p.chunk]. If the context is insufficient, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}")
    return {"answer": llm_fn(prompt), "citations": hits}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--docs", type=Path, required=True)
    b.add_argument("--index", type=Path, default=Path("index"))
    q = sub.add_parser("ask")
    q.add_argument("--index", type=Path, default=Path("index"))
    q.add_argument("--q", type=str, required=True)
    args = ap.parse_args()

    if args.cmd == "build":
        build(args.docs, args.index)
    else:
        print(json.dumps(answer(args.index, args.q), indent=2))


if __name__ == "__main__":
    main()
