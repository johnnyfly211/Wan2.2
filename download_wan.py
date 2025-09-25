from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Wan-AI/Wan2.2-Animate-14B",
    local_dir="Wan2.2-Animate-14B",
    resume_download=True,
    local_dir_use_symlinks=False,
    max_workers=1,
)
