# models/

    clinical_model.gguf          any instruction-tuned GGUF chat model (e.g. a 3-8B model, Q4_K_M) - override with UCG_LLM_MODEL_PATH
    embeddings/                  pre-downloaded embedding model files (fastembed cache or sentence-transformers folder)

Nothing here is downloaded at runtime. On a machine with internet, fetch the files once and copy them in:

    # fastembed (ONNX): run once online, then copy ~/.cache or point UCG_EMBEDDER_DIR at the cache
    python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='models/embeddings')"

If the files are missing the system still works: embeddings fall back to the hashing embedder (weaker semantic
search) and the LLM falls back to deterministic mode (templates + rules). `python cli.py health` shows what is active.

IMPORTANT: if you change the embedding model, rebuild the index:  python -m etl.run_etl --source <ucg_2023.db>
The index records the embedder name and semantic search refuses to run on a mismatch.
