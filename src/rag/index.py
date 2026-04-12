import pickle
import re
from pathlib import Path

import numpy as np

CORPUS_DIR = Path(__file__).parent / "corpus"
STORE_DIR = Path(__file__).parent / "index_store"
INDEX_FILE = STORE_DIR / "faiss.index"
META_FILE = STORE_DIR / "meta.pkl"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _read_chunks():
    chunks = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        sections = re.split(r"\n## ", text)
        first = sections[0]
        title_match = re.match(r"^#\s+(.+)", first.strip())
        doc_title = title_match.group(1).strip() if title_match else path.stem
        for section in sections[1:]:
            lines = section.strip().split("\n", 1)
            heading = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            if not body:
                continue
            chunks.append({
                "text": f"{heading}. {body}",
                "heading": heading,
                "doc": doc_title,
                "source": path.name,
            })
    return chunks


def _get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_NAME)


def build_index():
    import faiss

    chunks = _read_chunks()
    if not chunks:
        raise RuntimeError("No corpus chunks found")

    embedder = _get_embedder()
    texts = [c["text"] for c in chunks]
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    with open(META_FILE, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks


def load_index():
    import faiss

    if not INDEX_FILE.exists() or not META_FILE.exists():
        return build_index()

    index = faiss.read_index(str(INDEX_FILE))
    with open(META_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks
