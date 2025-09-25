#run_generate_fast.py
import importlib, inspect
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import os
os.environ["WAN_USE_FLASH_ATTN"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
import torch
import torch.nn.functional as F
from PIL import Image

# Enable Flash SDPA and fallbacks (PyTorch 2.5 auto-selects the best kernel)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

print("device:", torch.cuda.get_device_name(0),
      "cap:", torch.cuda.get_device_capability(0))
print("flash flag:", torch.backends.cuda.flash_sdp_enabled())

# Probe: call SDPA normally; PyTorch will use flash if shapes/dtype allow
q = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.float16)
k = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.float16)
v = torch.randn(1, 8, 512, 64, device="cuda", dtype=torch.float16)
_ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
torch.cuda.synchronize()
print("✅ SDPA ran (flash if compatible, else fallback)")


# Optional: prefer MAGMA for QR/SVD (avoids cuSOLVER hiccups on Windows)
try:
    torch.backends.cuda.preferred_linalg_library("magma")
except Exception:
    pass

# Ada/4090 matmul speedup (no quality change)
torch.set_float32_matmul_precision("high")

# ---------------------------------------------

from decord import VideoReader
try:
    import moviepy.editor as mpy
except:
    import moviepy as mpy  # fallback alias
    
    

CKPT_DIR = Path(r'.\Wan2.2-Animate-14B')
SRC_ROOT = Path(r'.\examples\wan_animate\animate\process_results_fasttok')
OUT_DIR  = Path(r'.\examples\wan_animate\animate\outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

pose = VideoReader(str(SRC_ROOT / 'src_pose.mp4'))
face = VideoReader(str(SRC_ROOT / 'src_face.mp4'))
print("[probe] pose frames:", len(pose), "face frames:", len(face))
f0 = pose[0].asnumpy(); g0 = face[0].asnumpy()
print("[probe] pose0 mean:", float(f0.mean()), "face0 mean:", float(g0.mean()))


pose_path = SRC_ROOT / 'src_pose.mp4'
vr = VideoReader(str(pose_path))
F = len(vr)   # total frames
fps = int(round(vr.get_avg_fps()))
clips = (F - 1 + (29 - 1)) // 28   # since clip_len=29 → stride=28

print(f"Video has {F} frames @ {fps} fps")
print(f"With clip_len=29, that will be ~{clips} clips")

cfgmod = importlib.import_module('wan.configs.wan_animate_14B')

def _looks_like_cfg(obj):
    need = ['t5_checkpoint','t5_tokenizer','clip_checkpoint','clip_tokenizer','vae_checkpoint']
    return all(hasattr(obj, k) for k in need)

CONFIG = None
for name in ['config','Config','CONFIG']:
    if hasattr(cfgmod, name) and _looks_like_cfg(getattr(cfgmod, name)):
        CONFIG = getattr(cfgmod, name); print(f'[cfg] using {name}'); break
if CONFIG is None:
    for name, obj in vars(cfgmod).items():
        if not name.startswith('_') and _looks_like_cfg(obj):
            CONFIG = obj; print(f'[cfg] using discovered {name}'); break
if CONFIG is None:
    CONFIG = SimpleNamespace(
        text_len=77, t5_dtype=torch.float16, param_dtype=torch.float16,
        num_train_timesteps=1000, prompt="", sample_neg_prompt="",
        t5_checkpoint='t5_encoder.pt', t5_tokenizer='t5_tokenizer',
        clip_checkpoint='clip_model.pt', clip_tokenizer='clip_tokenizer',
        vae_checkpoint='vae.pt', lora_checkpoint='relight_lora.pt',
    )
    print('[cfg] using fallback SimpleNamespace (edit filenames if needed)')

for k, v in dict(param_dtype=torch.float16, t5_dtype=torch.float16,
                 prompt="", sample_neg_prompt="", num_train_timesteps=1000,
                 text_len=77).items():
    if not hasattr(CONFIG, k): setattr(CONFIG, k, v)

from wan.animate import WanAnimate

model = WanAnimate(
    config=CONFIG, checkpoint_dir=str(CKPT_DIR),
    device_id=0, rank=0,
    t5_fsdp=False, dit_fsdp=False, use_sp=False,
    t5_cpu=False, init_on_cpu=True, convert_model_dtype=False,
    use_relighting_lora=False,
)

# --- VAE self-test: encode then decode the reference frame ---
from PIL import Image
import numpy as np, torch

ref_path = SRC_ROOT / 'src_ref.png'
img = Image.open(ref_path).convert('RGB')
arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0  # [-1,1]

# Encode expects (C,T,H,W) per item. We'll make T=1.
ref_chw  = torch.from_numpy(arr).permute(2, 0, 1).to('cuda', dtype=torch.bfloat16)  # (3,H,W)
ref_chtw = ref_chw.unsqueeze(1)  # (3,1,H,W)

with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    lat = model.vae.encode([ref_chtw])[0]      # latent

# Decode can return different container/shape variants across branches.
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    dec = model.vae.decode([lat])

# Normalize to a tensor
if isinstance(dec, (list, tuple)):
    dec = dec[0]
dec = torch.as_tensor(dec)

# We want (C,H,W). Handle common cases:
# - (C,1,H,W)          -> squeeze T
# - (1,C,1,H,W)        -> squeeze B and T
# - (1,C,H,W)          -> squeeze B
# - (C,T,H,W), T>=1    -> take T=0
# - already (C,H,W)    -> leave as is
if dec.dim() == 5:                 # (B,C,T,H,W)
    if dec.shape[0] == 1: dec = dec[0]          # -> (C,T,H,W)
if dec.dim() == 4:
    # (C,T,H,W) or (1,C,H,W) or (C,1,H,W)
    if dec.shape[1] >= 1 and dec.shape[-1] >= 8 and dec.shape[-2] >= 8:
        # likely (C,T,H,W)
        if dec.shape[1] == 1:
            dec = dec[:, 0]                     # -> (C,H,W)
        else:
            dec = dec[:, 0]                     # take first T -> (C,H,W)
    if dec.dim() == 4 and dec.shape[0] == 1:    # (1,C,H,W)
        dec = dec.squeeze(0)                    # -> (C,H,W)

# Final guard: if still has any singleton dims in the first two axes, squeeze them
while dec.dim() > 3 and 1 in dec.shape[:2]:
    dec = dec.squeeze(0)

assert dec.dim() == 3 and dec.shape[0] in (1, 3), f"Unexpected decode shape: {tuple(dec.shape)}"
if dec.shape[0] == 1:
    dec = dec.repeat(3, 1, 1)                   # grayscale -> RGB

rec_img = dec.clamp(-1, 1).add(1).mul(127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
Image.fromarray(rec_img).save(OUT_DIR / 'vae_roundtrip_ref.png')
print("[check] wrote:", OUT_DIR / 'vae_roundtrip_ref.png')


videos = model.generate(
    src_root_path=str(SRC_ROOT),
    replace_flag=False,
    clip_len=33,      # very small
    refert_num=1,
    sample_solver='dpm++',
    sampling_steps=10,
    guide_scale=1,
    seed=42,
    input_prompt="a person grabbing their chest",
    n_prompt="low quality, blurry",
    offload_model=True,
)

# --- Convert model output to frames safely (handles CNHW/NCHW/NHWC) ---
print("videos.shape:", tuple(videos.shape), "min/max:", float(videos.min()), float(videos.max()))
vid = videos.detach().to(torch.float32).clamp(-1, 1)  # [-1,1]

if vid.dim() != 4:
    raise RuntimeError(f"Unexpected video tensor rank: {vid.dim()} (shape={tuple(vid.shape)})")

if vid.shape[0] in (1, 3, 4) and vid.shape[1] >= 1:
    # CNHW  -> NHWC
    # e.g. (3, N, H, W)
    frames = (vid.add(1).mul(127.5)).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    layout = "CNHW→NHWC"
elif vid.shape[1] in (1, 3, 4):
    # NCHW  -> NHWC
    frames = (vid.add(1).mul(127.5)).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    layout = "NCHW→NHWC"
elif vid.shape[-1] in (1, 3, 4):
    # NHWC already
    frames = (vid.add(1).mul(127.5)).to(torch.uint8).cpu().numpy()
    layout = "NHWC"
else:
    raise RuntimeError(f"Unexpected video tensor shape: {tuple(vid.shape)}")

print(f"[layout] converted using: {layout}; frames array shape: {frames.shape}")  # (N,H,W,C)

# Optional quick sanity check
Image.fromarray(frames[0]).save(OUT_DIR / "debug_first_frame.png")


# --- FPS (from src_pose if available) ---
fps = 20
try:
    vr = VideoReader(str(SRC_ROOT / 'src_pose.mp4'))
    fps = max(1, int(round(vr.get_avg_fps())))
except Exception:
    pass

# --- Write video ---
out_file = OUT_DIR / 'wan_animate_fast.mp4'
mpy.ImageSequenceClip(list(frames), fps=fps).write_videofile(str(out_file))
print('Saved:', out_file)

