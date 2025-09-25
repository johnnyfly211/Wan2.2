# run_generate_cached.py
import argparse, os, sys, time, json
import torch
import torchvision
from einops import rearrange
from omegaconf import OmegaConf

# --------------------------
# Paths (adjust if needed)
# --------------------------
PROJECT_ROOT   = os.path.abspath(os.path.dirname(__file__))
MODEL_ROOT     = os.path.join(PROJECT_ROOT, "Wan2.2-Animate-14B")  # <-- your model dir
SEARCH_DIRS    = [
    os.path.join(PROJECT_ROOT, "examples", "wan_animate", "replace"),
    os.path.join(PROJECT_ROOT, "examples", "wan_animate"),
]

# Add Wan2.2 to sys.path
sys.path.insert(0, PROJECT_ROOT)
from wan.animate import WanAnimate


# --------------------------
# Helpers
# --------------------------
def _is_valid_plain(d):
    need = ["src_pose.mp4", "src_face.mp4", "src_ref.png"]
    return all(os.path.isfile(os.path.join(d, f)) for f in need)

def _is_valid_replace(d):
    need = ["src_pose.mp4", "src_face.mp4", "src_ref.png", "src_bg.mp4", "src_mask.mp4"]
    return all(os.path.isfile(os.path.join(d, f)) for f in need)

def _find_latest_src():
    candidates = []
    for base in SEARCH_DIRS:
        if not os.path.isdir(base): continue
        for name in os.listdir(base):
            if not name.startswith("process_results_"): continue
            path = os.path.join(base, name)
            if not os.path.isdir(path): continue
            mtime = os.path.getmtime(path)
            candidates.append((mtime, path))
    candidates.sort(reverse=True)  # newest first
    for _, p in candidates:
        if _is_valid_replace(p): return p, True
    for _, p in candidates:
        if _is_valid_plain(p): return p, False
    return None, None

def _normalize_src(p):
    if os.path.isdir(p) and os.path.basename(p).startswith("process_results_"):
        return p
    inside = [x for x in os.listdir(p) if x.startswith("process_results_") and os.path.isdir(os.path.join(p, x))]
    if len(inside) == 1:
        return os.path.join(p, inside[0])
    return p


def _load_config(model_root):
    # Candidate paths
    candidates = [
        os.path.join(model_root, "process_checkpoint", "config.yaml"),
        os.path.join(model_root, "process_checkpoint", "config.yml"),
        os.path.join(model_root, "config.yaml"),
        os.path.join(model_root, "config.yml"),
        os.path.join(model_root, "config.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            if p.endswith((".yaml", ".yml")):
                print(f"[OK] Using config: {p}")
                return OmegaConf.load(p)
            elif p.endswith(".json"):
                with open(p, "r", encoding="utf-8") as f:
                    return OmegaConf.create(json.load(f))
    tried = "\n  - ".join(candidates)
    raise FileNotFoundError(f"Could not find config in:\n  - {tried}")


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser(description="Run Wan2.2 Animate-14B from cached preprocess outputs.")
    ap.add_argument("--src", type=str, default=None, help="Path to a process_results_* folder. If omitted, auto-pick newest.")
    ap.add_argument("--clip_len", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--solver", type=str, default="dpm++", choices=["dpm++", "unipc"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guide", type=float, default=1.0)
    ap.add_argument("--prompt", type=str, default="a person talking naturally")
    ap.add_argument("--neg", type=str, default="")
    args = ap.parse_args()

    # --- Find SRC folder
    if args.src:
        src_root = _normalize_src(os.path.abspath(args.src))
        replace_flag = _is_valid_replace(src_root)
        if not (replace_flag or _is_valid_plain(src_root)):
            raise SystemExit(f"[ERR] {src_root} missing required src files.")
    else:
        src_root, replace_flag = _find_latest_src()
        if not src_root:
            raise SystemExit("[ERR] No valid process_results_* folder found.")
    print(f"[OK] Using SRC_ROOT = {src_root}")
    print(f"[OK] Mode: {'replace' if replace_flag else 'plain'}")

    # --- Load config
    cfg = _load_config(MODEL_ROOT)

    # --- Init model
    model = WanAnimate(
        config=cfg,
        checkpoint_dir=MODEL_ROOT,
        device_id=0,
        rank=0,
        t5_cpu=False,
        dit_fsdp=False,
        use_sp=False,
        init_on_cpu=False,
        convert_model_dtype=False,
        use_relighting_lora=False,
    )

    # --- Generate
    videos = model.generate(
        src_root_path=src_root,
        replace_flag=replace_flag,
        clip_len=args.clip_len,
        refert_num=1,
        shift=5.0,
        sample_solver=args.solver,
        sampling_steps=args.steps,
        guide_scale=args.guide,
        input_prompt=args.prompt,
        n_prompt=args.neg,
        seed=args.seed,
        offload_model=True,
    )

    # --- Save
    out_dir  = os.path.join(PROJECT_ROOT, "examples", "wan_animate", "replace", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"cached_out_{int(time.time())}.mp4")

    vid = ((rearrange(videos, "c t h w -> t h w c").cpu().clamp(-1,1) + 1.0) * 127.5).byte()
    torchvision.io.write_video(out_path, vid, fps=20, video_codec="h264")
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
