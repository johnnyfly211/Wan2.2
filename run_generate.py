import importlib
import inspect
from types import SimpleNamespace
from pathlib import Path
import torch

# --- paths ---
CKPT_DIR = Path(r'.\Wan2.2-Animate-14B')
SRC_ROOT = Path(r'.\examples\wan_animate\replace\process_results')  # has src_pose/bg/mask/face + src_ref.png
OUT_DIR  = Path(r'.\examples\wan_animate\replace\outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- 1) find/load a config object from wan_animate_14B ---
cfgmod = importlib.import_module('wan.configs.wan_animate_14B')

def _looks_like_cfg(obj):
    need = ['t5_checkpoint','t5_tokenizer','clip_checkpoint','clip_tokenizer','vae_checkpoint']
    return all(hasattr(obj, k) for k in need)

CONFIG = None

# A) try common names
for name in ['config','Config','CONFIG']:
    if hasattr(cfgmod, name):
        candidate = getattr(cfgmod, name)
        if _looks_like_cfg(candidate):
            CONFIG = candidate
            print(f'[cfg] using attribute {name} from wan_animate_14B')
            break

# B) try callables that return a config
if CONFIG is None:
    for name, obj in vars(cfgmod).items():
        if callable(obj) and any(s in name.lower() for s in ['get_config','make_config','build_config','config']):
            try:
                ret = obj()
                if _looks_like_cfg(ret):
                    CONFIG = ret
                    print(f'[cfg] using callable {name}() from wan_animate_14B')
                    break
            except TypeError:
                # try calling with ckpt dir if it wants a path
                try:
                    ret = obj(str(CKPT_DIR))
                    if _looks_like_cfg(ret):
                        CONFIG = ret
                        print(f'[cfg] using callable {name}(CKPT_DIR) from wan_animate_14B')
                        break
                except Exception:
                    pass
            except Exception:
                pass

# C) scan module for any object that has needed attrs
if CONFIG is None:
    for name, obj in vars(cfgmod).items():
        if not name.startswith('_') and not inspect.ismodule(obj) and _looks_like_cfg(obj):
            CONFIG = obj
            print(f'[cfg] using discovered object {name} from wan_animate_14B')
            break

# D) last resort: build a minimal config (filenames must match your checkpoint bundle)
if CONFIG is None:
    # If you know the exact filenames in your checkpoint folder, set them here:
    # e.g. 't5_checkpoint': 't5_encoder.pt', etc.
    CONFIG = SimpleNamespace(
        # model hyper params (sane defaults)
        text_len=77,
        t5_dtype=torch.float16,
        param_dtype=torch.float16,
        num_train_timesteps=1000,
        prompt="",
        sample_neg_prompt="",
        # filenames under CKPT_DIR (edit if different!)
        t5_checkpoint='t5_encoder.pt',
        t5_tokenizer='t5_tokenizer',
        clip_checkpoint='clip_model.pt',
        clip_tokenizer='clip_tokenizer',
        vae_checkpoint='vae.pt',
        # optional:
        lora_checkpoint='relight_lora.pt',
    )
    print('[cfg] using fallback SimpleNamespace CONFIG (edit filenames if they differ)')

# --- 2) ensure required fields exist (add harmless defaults if missing) ---
for k, v in dict(
    param_dtype=torch.float16,
    t5_dtype=torch.float16,
    prompt="",
    sample_neg_prompt="",
    num_train_timesteps=1000,
    text_len=77,
).items():
    if not hasattr(CONFIG, k):
        setattr(CONFIG, k, v)

# --- 3) run generation ---
from wan.animate import WanAnimate

model = WanAnimate(
    config=CONFIG,
    checkpoint_dir=str(CKPT_DIR),
    device_id=0,
    rank=0,
    t5_fsdp=False,
    dit_fsdp=False,
    use_sp=False,
    t5_cpu=False,          # you have GPU, keep encoder on GPU if possible
    init_on_cpu=True,      # Wan uses CPU init then moves to GPU unless sharded
    convert_model_dtype=False,
    use_relighting_lora=False,
)

out_path = model.generate(
    src_root_path=str(SRC_ROOT),
    replace_flag=True,         # were using the replace pipeline
    sample_solver='dpm++',
    sampling_steps=28,
    guide_scale=2.5,
    seed=42,
    input_prompt="",           # optional text prompt tweaks
    n_prompt="",               # optional negative prompt
    offload_model=True,        # free GPU memory between steps where possible
)

print(' Done. Output video:', out_path)
