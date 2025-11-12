# SAMRetriever — Documentation

## Summary of recent changes
- Added Faiss index metadata file used for tests and offline indexing.
- Metadata includes `image_paths`, `config` (model_name, embedding_dim, top_k) and `stats` (totals and running averages).
- SAMRetriever now stores:
  - `global_embeddings` — one vector per image (N, D)
  - `region_embeddings` — per-region vectors (M, D) saved/loaded as numeric ndarrays
  - `region_image_paths` — parallel list/array mapping each region to its source image path

## Why this matters
- Faiss ids must map 1:1 to `image_paths` order when building the index.
- Embedding dimensionality (`embedding_dim`) in metadata must match vectors added to Faiss.
- Keeping both global and region embeddings allows:
  - fast global retrieval (Faiss on global embeddings)
  - region-level reranking (MaxSim) using region embeddings

## File location
- Metadata: `./test_faiss_index_metadata.json`
- Index and embeddings expected under the repo `data/` directory (repo-relative paths stored in metadata).

## Metadata schema (important fields)
- `image_paths`: ordered list of image paths (order is significant).
- `config`:
  - `model_name`: CLIP model used for embedding generation.
  - `embedding_dim`: integer (e.g., 512). Must match produced vectors.
  - `top_k`: default number of neighbours to return.
- `stats`:
  - `total_embeddings_generated`, `total_queries_processed`
  - `avg_encoding_time`, `avg_retrieval_time` (running averages)

## Initialization and saving notes (important)
- region_embeddings must be converted to a numeric 2‑D ndarray before saving:
  - do not save Python lists of arrays (object dtype) — convert via `np.vstack` or pad to (N, max_regions, D).
- When loading, defensively convert object arrays back to numeric ndarrays:
  - use `allow_pickle=True` when necessary, then `np.vstack(loaded)` if `loaded.dtype == object`.
- Provide a backwards-compatible alias `self.embeddings = self.global_embeddings` if other code expects `embeddings`.

## Quick usage examples

- Load metadata:
```py
import json
with open("test_faiss_index_metadata.json","r") as f:
    meta = json.load(f)
image_paths = meta["image_paths"]
d = meta["config"]["embedding_dim"]
```

- Build Faiss index (IndexFlatL2 example):
```py
import faiss, numpy as np
index = faiss.IndexFlatL2(d)
index.add(embeddings.astype(np.float32))  # embeddings shape (N, d)
```

- Query and map ids back to images:
```py
D, I = index.search(query_vectors.astype(np.float32), k)
results = [image_paths[idx] for idx in I[0]]
```

- Save region embeddings safely:
```py
# padded shape: (N, max_regions, D) or stacked shape (M, D)
np.save("data/embeddings/region_embeddings.npy", region_embeddings.astype(np.float32))
```

## Best practices / recommendations
- Always validate embedding shapes (D matches metadata) before adding to Faiss.
- Keep `image_paths` order consistent with how embeddings are added.
- Cap regions per image (e.g., `max_regions_per_image`) to limit memory and file size.
- Use padding + mask for variable-length region sets if region-level arrays must be saved as a 3D tensor.
- When debugging, enable INFO logs for encode/retrieve entry and exit points; avoid relying on DEBUG-only messages.

## Troubleshooting hints
- If retrieval errors mention "object" dtype or reduction errors, inspect saved `region_embeddings` with `np.load(..., allow_pickle=True)` and convert to numeric arrays.
- If Faiss `search` raises shape errors, ensure the query vector has shape `(1, D)` (use `np.expand_dims(query, 0)`).

## Next steps / automation ideas
- Add a small utility script to:
  1. Load metadata
  2. Generate embeddings with a given CLIP model
  3. Build Faiss index
  4. Save/update metadata stats
- Add CI checks to validate that `embedding_dim` in metadata equals the vectors' second dimension before indexing.