# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os
import warnings
import torch
import torch.nn.functional as F

# Optional toggles (used by attention(); flash_attention ignores these and just falls back if needed)
USE_FLASH = os.getenv("WAN_USE_FLASH_ATTN", "0") == "1"
FORCE_MATH_SDP = os.getenv("WAN_SDP_FORCE_MATH", "0") == "1"

# Probe FlashAttention availability
try:
    import flash_attn_interface as _fa3
    FLASH_ATTN_3_AVAILABLE = True
except Exception:
    _fa3 = None
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn as _fa2
    FLASH_ATTN_2_AVAILABLE = True
except Exception:
    _fa2 = None
    FLASH_ATTN_2_AVAILABLE = False

__all__ = ["flash_attention", "attention"]


def _make_cu_seqlens(lens: torch.Tensor) -> torch.Tensor:
    return torch.cat([lens.new_zeros(1), lens], dim=0).cumsum(0, dtype=torch.int32)


def _sdpa_fallback(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    q_scale=None,
    causal: bool = False,
    dtype=torch.bfloat16,
):
    # Inputs: [B, L, H, C]  → SDPA expects [B, H, L, C]
    if q_scale is not None:
        q = q * q_scale

    q = q.transpose(1, 2).to(dtype)  # [B,H,Lq,C]
    k = k.transpose(1, 2).to(dtype)  # [B,H,Lk,C]
    v = v.transpose(1, 2).to(dtype)  # [B,H,Lk,C]

    B, H, Lq, _ = q.shape
    _, _, Lk, _ = k.shape

    # Build a boolean mask only if variable lengths were provided
    base_mask = None
    if q_lens is not None or k_lens is not None:
        if q_lens is None:
            q_lens = torch.full((B,), Lq, dtype=torch.int32, device=q.device)
        if k_lens is None:
            k_lens = torch.full((B,), Lk, dtype=torch.int32, device=k.device)
        ar_q = torch.arange(Lq, device=q.device)
        ar_k = torch.arange(Lk, device=k.device)
        masks = []
        for b in range(B):
            rq = ar_q >= int(q_lens[b])
            rk = ar_k >= int(k_lens[b])
            masks.append(rq[:, None] | rk[None, :])   # True = masked
        base_mask = torch.stack(masks, dim=0)         # [B, Lq, Lk]

    # Choose a chunk size for queries (smaller → lower peak VRAM, slower)
    # You can tune via env; default 256 works well on 24 GB cards.
    CHUNK = int(os.getenv("WAN_SDP_CHUNK", "256"))

    outs = []
    for qs in range(0, Lq, CHUNK):
        qe = min(Lq, qs + CHUNK)               # [qs:qe]
        q_chunk = q[:, :, qs:qe, :]            # [B,H,QL,C]

        if base_mask is not None:
            attn_mask = base_mask[:, qs:qe, :]           # [B,QL,Lk]
            attn_mask = attn_mask[:, None].expand(B, H, qe - qs, Lk)  # [B,H,QL,Lk]
            attn_mask = attn_mask.reshape(B * H, qe - qs, Lk)
        else:
            attn_mask = None

        # Prefer mem-efficient kernel; fall back gracefully if not available
        try:
            from torch.nn.attention import sdpa_kernel as _sdpa_kernel
            ctx = _sdpa_kernel(use_flash=False, use_mem_efficient=True, use_math=False)
            use_ctx = True
        except Exception:
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=True)
            use_ctx = True

        if use_ctx:
            with ctx:
                try:
                    out_chunk = F.scaled_dot_product_attention(
                        q_chunk, k, v,
                        attn_mask=attn_mask,
                        is_causal=causal,
                        dropout_p=0.0,
                        scale=softmax_scale,   # supported on PyTorch ≥ 2.5
                    )
                except TypeError:
                    out_chunk = F.scaled_dot_product_attention(
                        q_chunk, k, v,
                        attn_mask=attn_mask,
                        is_causal=causal,
                        dropout_p=0.0,
                    )
        else:
            out_chunk = F.scaled_dot_product_attention(
                q_chunk, k, v,
                attn_mask=attn_mask,
                is_causal=causal,
                dropout_p=0.0,
            )

        outs.append(out_chunk)  # [B,H,QL,C]

        # Help the allocator between chunks
        del out_chunk, attn_mask, q_chunk
        torch.cuda.empty_cache()

    out = torch.cat(outs, dim=2)  # [B,H,Lq,C]
    return out.transpose(1, 2).contiguous()  # [B,L,H,C]


def flash_attention(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    q_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    deterministic: bool = False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    Fast path using FlashAttention v3/v2 when available.
    If FA is not available, this function auto-falls back to SDPA — no errors.
    Shapes: q=[B, Lq, Hq, C], k=[B, Lk, Hk, C], v=[B, Lk, Hk, C2]
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    if q.device.type != "cuda" or dtype not in half_dtypes or q.size(-1) > 256:
        # Non-CUDA / unsupported head_dim → SDPA fallback
        return _sdpa_fallback(q, k, v, q_lens, k_lens, 0.0, softmax_scale, q_scale, causal, dtype)

    if q_scale is not None:
        q = q * q_scale

    B, Lq, Hq, C = q.shape
    _, Lk, Hk, _ = k.shape
    out_dtype = q.dtype

    def _to_half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # Flatten var-len
    if q_lens is None:
        q_lens = torch.full((B,), Lq, dtype=torch.int32, device=q.device)
        q_flat = _to_half(q.flatten(0, 1))
    else:
        q_flat = _to_half(torch.cat([q[b, :q_lens[b]] for b in range(B)], dim=0))

    if k_lens is None:
        k_lens = torch.full((B,), Lk, dtype=torch.int32, device=k.device)
        k_flat = _to_half(k.flatten(0, 1))
        v_flat = _to_half(v.flatten(0, 1))
    else:
        k_flat = _to_half(torch.cat([k[b, :k_lens[b]] for b in range(B)], dim=0))
        v_flat = _to_half(torch.cat([v[b, :k_lens[b]] for b in range(B)], dim=0))

    if k_flat.dtype != q_flat.dtype:
        k_flat = k_flat.to(q_flat.dtype)
    if v_flat.dtype != q_flat.dtype:
        v_flat = v_flat.to(q_flat.dtype)

    # Try FA3 then FA2; otherwise fall back to SDPA (no error)
    if (version in (None, 3)) and FLASH_ATTN_3_AVAILABLE:
        try:
            out = _fa3.flash_attn_varlen_func(
                q=q_flat,
                k=k_flat,
                v=v_flat,
                cu_seqlens_q=_make_cu_seqlens(q_lens).to(q.device, non_blocking=True),
                cu_seqlens_k=_make_cu_seqlens(k_lens).to(q.device, non_blocking=True),
                seqused_q=None,
                seqused_k=None,
                max_seqlen_q=int(Lq),
                max_seqlen_k=int(Lk),
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic,
            )[0].unflatten(0, (B, Lq))
            return out.to(out_dtype)
        except Exception:
            pass

    if FLASH_ATTN_2_AVAILABLE:
        try:
            out = _fa2.flash_attn_varlen_func(
                q=q_flat,
                k=k_flat,
                v=v_flat,
                cu_seqlens_q=_make_cu_seqlens(q_lens).to(q.device, non_blocking=True),
                cu_seqlens_k=_make_cu_seqlens(k_lens).to(q.device, non_blocking=True),
                max_seqlen_q=int(Lq),
                max_seqlen_k=int(Lk),
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            ).unflatten(0, (B, Lq))
            return out.to(out_dtype)
        except Exception:
            pass

    # Final safety net: SDPA
    return _sdpa_fallback(q, k, v, q_lens, k_lens, 0.0, softmax_scale, q_scale, causal, dtype)


def attention(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    q_scale=None,
    causal: bool = False,
    window_size=(-1, -1),
    deterministic: bool = False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """
    General entry point: uses FlashAttention if available *and* WAN_USE_FLASH_ATTN=1,
    else SDPA. (flash_attention() already auto-falls back; this just honors the env toggle.)
    """
    if USE_FLASH and (FLASH_ATTN_3_AVAILABLE or FLASH_ATTN_2_AVAILABLE):
        try:
            return flash_attention(
                q, k, v,
                q_lens=q_lens, k_lens=k_lens,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
                dtype=dtype,
                version=fa_version,
            )
        except Exception:
            pass
    return _sdpa_fallback(q, k, v, q_lens, k_lens, 0.0, softmax_scale, q_scale, causal, dtype)
