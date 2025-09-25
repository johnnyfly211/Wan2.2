# =========================
# run_wan_4090.ps1
# RTX 4090 • SDPA • FP16 • Robust alignment
# =========================

$ErrorActionPreference = "Stop"

# --- Project root ---
$ROOT = "D:\A-Coding Projects\Wan2.2"
Set-Location $ROOT

# --- Env (VRAM-friendly & debug) ---
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:64,expandable_segments:True"
$env:WAN_USE_FLASH_ATTN      = "0"      # SDPA path unless you have flash-attn
$env:WAN_SDP_CHUNK           = "128"    # adjust 96..160 if needed
$env:NVIDIA_TF32_OVERRIDE    = "0"
$env:WAN_DEBUG_ALIGN         = "1"      # set "0" to quiet alignment logs
$env:WAN_DEBUG               = "0"      # set "1" for verbose sampler logs

# --- I/O ---
$CKPT_DIR = ".\Wan2.2-Animate-14B"
$VIDEO    = ".\WIN_20250924_09_43_14_Pro.mp4"
$REF_IMG  = ".\image.png"

# You can change this; runtime will enforce multiples of 8 anyway.
$RES_W = 896
$RES_H = 512

$SAVE_DIR = ".\examples\wan_animate\replace\process_results_${RES_W}x${RES_H}"

# Clean + make preprocess dir
Remove-Item -Recurse -Force $SAVE_DIR -ErrorAction SilentlyContinue
New-Item -ItemType Directory $SAVE_DIR -Force | Out-Null

# --- Preprocess ---
poetry run python .\wan\modules\animate\preprocess\preprocess_data.py `
  --ckpt_path "$CKPT_DIR\process_checkpoint" `
  --video_path "$VIDEO" `
  --refer_path "$REF_IMG" `
  --save_path "$SAVE_DIR" `
  --resolution $RES_W $RES_H `
  --fps 20 `
  --iterations 2 --k 5 --w_len 1 --h_len 1 `
  --replace_flag

# --- Generate (FP16) ---
$genPy = @"
import os, importlib
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import amp

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64,expandable_segments:True")
os.environ.setdefault("WAN_USE_FLASH_ATTN", "0")
os.environ.setdefault("WAN_SDP_CHUNK", "128")
torch.set_float32_matmul_precision("high")

try:
    from decord import VideoReader
except Exception:
    VideoReader = None
try:
    import moviepy.editor as mpy
except Exception:
    import moviepy as mpy  # fallback alias

CKPT_DIR = Path(r'.\Wan2.2-Animate-14B')
SRC_ROOT = Path(r'$SAVE_DIR')
OUT_DIR  = Path(r'.\examples\wan_animate\replace\outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    print('[cfg] using fallback SimpleNamespace')

for k, v in dict(param_dtype=torch.float16, t5_dtype=torch.float16,
                 prompt="", sample_neg_prompt="", num_train_timesteps=1000,
                 text_len=77).items():
    if not hasattr(CONFIG, k): setattr(CONFIG, k, v)
CONFIG.param_dtype = torch.float16
if hasattr(CONFIG, "t5_dtype"): CONFIG.t5_dtype = torch.float16

from wan.animate import WanAnimate
model = WanAnimate(
    config=CONFIG, checkpoint_dir=str(CKPT_DIR),
    device_id=0, rank=0,
    t5_fsdp=False, dit_fsdp=False, use_sp=False,
    t5_cpu=False, init_on_cpu=True, convert_model_dtype=False,
    use_relighting_lora=False,
)

try:
    clip_mod = getattr(model.clip, "model", model.clip)
    clip_mod.to(dtype=torch.float16, memory_format=torch.contiguous_format)
    model.clip.dtype = torch.float16
except Exception as e:
    print("[warn] couldn't force CLIP fp16/contig:", e)

print(f"[Generate] target {${RES_W}}x${RES_H}, clip_len=32, steps=10 (runtime will auto-align as needed)")

with amp.autocast(device_type="cuda", dtype=torch.float16):
    videos = model.generate(
        src_root_path=str(SRC_ROOT),
        replace_flag=True,
        clip_len=32,
        refert_num=1,
        sample_solver='dpm++',
        sampling_steps=10,
        guide_scale=1.0,
        seed=42,
        input_prompt="",
        n_prompt="",
        offload_model=True,
    )

C, N, H, W = videos.shape
frames = (videos.clamp(-1,1).add(1).mul(127.5)).to(torch.uint8).permute(1,2,3,0).cpu().numpy()

fps = 20
pose_path = SRC_ROOT / 'src_pose.mp4'
if VideoReader is not None:
    try:
        vr = VideoReader(str(pose_path))
        fps = max(1, int(round(vr.get_avg_fps())))
    except Exception:
        pass

out_file = OUT_DIR / f'wan_animate_replace_{H}x{W}.mp4'
mpy.ImageSequenceClip(list(frames), fps=fps).write_videofile(str(out_file))
print('Saved:', out_file)
"@

Set-Content -Path ".\run_generate_4090.py" -Value $genPy -Encoding UTF8

poetry run python .\run_generate_4090.py
