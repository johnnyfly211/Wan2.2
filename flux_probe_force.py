# flux_probe_force.py
import os, torch
from pathlib import Path
from transformers import AutoTokenizer
from diffusers import FluxKontextPipeline

os.environ["TRANSFORMERS_TOKENIZER_FORCE_FAST"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

root = Path(r".\Wan2.2-Animate-14B\process_checkpoint\FLUX.1-Kontext-dev")
tok_dir = root / "tokenizer"
tok2_dir = root / "tokenizer_2"

print("[flux] repo:", root.resolve())
print("[flux] tokenizer.json present:", (tok_dir / "tokenizer.json").exists())

# Build fast tokenizers up-front
tok  = AutoTokenizer.from_pretrained(tok_dir,  use_fast=True)
tok2 = AutoTokenizer.from_pretrained(tok2_dir if tok2_dir.exists() else tok_dir, use_fast=True)

pipe = FluxKontextPipeline.from_pretrained(
    str(root),
    dtype=torch.bfloat16,     # 4090-friendly
    tokenizer=tok,            # inject, so it won't try to “convert slow”
    tokenizer_2=tok2          # if the pipeline uses a second tokenizer
)

print("[flux] components:", list(pipe.components.keys()))
print("[flux] tokenizer:", type(pipe.tokenizer).__name__)
print("[flux] dtype:", pipe.dtype)
print("[flux] ✅ loaded OK")
