import numpy as np
import streamlit as st

from .index import load_index, _get_embedder


@st.cache_resource(show_spinner=False)
def _get_index_and_meta():
    index, chunks = load_index()
    return index, chunks


@st.cache_resource(show_spinner=False)
def _cached_embedder():
    return _get_embedder()


def search(query, k=5):
    index, chunks = _get_index_and_meta()
    embedder = _cached_embedder()
    vec = embedder.encode([query], normalize_embeddings=True)
    vec = np.asarray(vec, dtype="float32")
    scores, idx = index.search(vec, k)
    results = []
    for score, i in zip(scores[0], idx[0]):
        if i < 0 or i >= len(chunks):
            continue
        hit = dict(chunks[i])
        hit["score"] = float(score)
        results.append(hit)
    return results
