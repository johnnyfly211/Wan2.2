# flux_probe.py
import torch, os, json
from diffusers import FluxKontextPipeline

path = r".\Wan2.2-Animate-14B\process_checkpoint\FLUX.1-Kontext-dev"
print("[flux] trying:", path)
print("[flux] exists:", os.path.isdir(path), "files:", len(os.listdir(path)) if os.path.isdir(path) else 0)

pipe = FluxKontextPipeline.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,   # bfloat16 is fine on 4090
    variant=None
)
# show components we loaded
print("[flux] components:", list(pipe.components.keys()))
print("[flux] tokenizer:", type(pipe.tokenizer).__name__)
print("[flux] dtype:", pipe.dtype)
print("[flux] ✅ loaded OK")
