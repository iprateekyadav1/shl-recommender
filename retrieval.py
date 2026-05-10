"""
Semantic retrieval layer over catalog.json using sentence-transformers + FAISS.
The index is built once at startup and reused for all queries.
"""

import logging
from typing import List, Dict, Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"


class CatalogRetriever:
    def __init__(self, catalog: List[Dict[str, Any]]) -> None:
        self._catalog = catalog
        self._model = SentenceTransformer(_MODEL_NAME)
        self._index = self._build_index()

    def _build_index(self) -> faiss.IndexFlatIP:
        """Embed all catalog descriptions and build a FAISS inner-product index."""
        logger.info("Building FAISS index for %d catalog items…", len(self._catalog))

        # Combine name + tags + description for richer embeddings
        texts = []
        for item in self._catalog:
            tags_str = " ".join(item.get("tags", []))
            composite = f"{item['name']}. {tags_str}. {item['description']}"
            texts.append(composite)

        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # Normalise so inner-product = cosine similarity
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))  # type: ignore[arg-type]
        logger.info("FAISS index built (dim=%d)", dim)
        return index

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Return top-k catalog items most semantically similar to *query*."""
        if not self._catalog:
            return []

        top_k = min(top_k, len(self._catalog))
        qvec = self._model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(qvec)

        scores, indices = self._index.search(qvec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self._catalog[idx])
            item["_score"] = float(score)
            results.append(item)
        return results
