# Third party sources

This project applies existing retrieval methods and pretrained checkpoints for the
INT4413 course, Team 12. It does not claim to invent BM25, BEIR, RRF, TAS-B,
MiniLM, BGE or Qwen.

- Upstream BEIR: <https://github.com/beir-cellar/beir>, Apache License 2.0.
  Paper-era checkout: `52de18ae6f1ccc7426eb70895c0e43e1552ad493`.
  The checkout is downloaded separately under ignored `upstream/beir/` and retains
  its own license. See `tools/setup_paper_checkout.py`.
- Pyserini baseline image: `beir/pyserini-fastapi`, pinned by digest in
  `run_paper_baseline.py`; attribution and runtime details are recorded in the audit.
- Python packages and model checkpoints retain their respective licenses. Their
  model weights and dependency code are not distributed in this repository.
- `legacy/v1/` preserves the original local project for provenance. Its TLS bypass
  and import-time network probes are historical behavior; use the V2 entrypoints.

Paper references and measured project results are stored separately. Historical
summary JSON files are preserved, but cannot be reverified without their rankings.
