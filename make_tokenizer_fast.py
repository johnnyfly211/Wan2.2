# make_tokenizer_fast.py
from transformers import AutoTokenizer
from pathlib import Path

tok_dir = Path(r".\Wan2.2-Animate-14B\process_checkpoint\FLUX.1-Kontext-dev\tokenizer")
print("[tok] loading slow/fast and saving tokenizer.json into:", tok_dir)

# Force a FAST tokenizer; if it doesn't exist, Transformers will convert it
tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
tok.save_pretrained(tok_dir)
print("[tok] wrote files:", list(p.name for p in tok_dir.iterdir()))