# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from torch import nn
import torch
import os
from typing import Tuple, Optional
from einops import rearrange
import torch.nn.functional as F
import math
from ...distributed.util import gather_forward, get_rank, get_world_size

WAN_USE_FLASH = os.getenv("WAN_USE_FLASH_ATTN", "0") == "1"

try:
    # Some repos expose this as flash_attn_func; others via flash_attn.flash_attn_func
    from flash_attn import flash_attn_func as _flash_attn_func  # type: ignore
except Exception:
    _flash_attn_func = None

MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    max_seqlen_q=None,
    batch_size=1,
):
    """
    Expect q,k,v shaped [B*, L, H, D] where B* may be a merged batch (e.g. (B*Lclips)).
    Returns [B*, L, H*D].
    """
    if mode == "torch":
        # SDPA path needs [B*, H, L, D]
        q_sdpa = q.transpose(1, 2)
        k_sdpa = k.transpose(1, 2)
        v_sdpa = v.transpose(1, 2)
        x = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=attn_mask, dropout_p=0.0, is_causal=causal)
        x = x.transpose(1, 2)  # -> [B*, L, H, D]

    elif mode == "flash":
        # Our helper returns [B*, L, H, D]
        x = _attention_flash_or_sdpa(q, k, v, causal=causal)
        # normalize shape explicitly (some callers pass batch_size/max_seqlen_q)
        if x.dim() == 4 and batch_size is not None and max_seqlen_q is not None:
            # If already [B*, L, H, D], this is a no-op reshape
            x = x.view(batch_size, max_seqlen_q, x.shape[-2], x.shape[-1])

    elif mode == "vanilla":
        scale = 1.0 / math.sqrt(q.size(-1))
        # q,k,v expected [B*, H, L, D] for matmul; transpose from [B*, L, H, D]
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        b, h, s, _ = qh.shape
        s1 = kh.size(2)
        bias = torch.zeros(b, h, s, s1, dtype=qh.dtype, device=qh.device)
        if causal:
            mask = torch.ones(b, h, s, s, dtype=torch.bool, device=qh.device).tril()
            bias.masked_fill_(~mask, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                bias.masked_fill_(~attn_mask, float("-inf"))
            else:
                bias += attn_mask
        attn = (qh @ kh.transpose(-2, -1)) * scale + bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = (attn @ vh).transpose(1, 2)  # -> [B*, L, H, D]
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    # --- ALWAYS return [B*, L, H*D] ---
    if x.dim() == 4:
        b, l, h, d = x.shape
        x = x.reshape(b, l, h * d)
    return x



def _attention_flash_or_sdpa(q, k, v, *, causal: bool, softmax_scale=None):
    """
    Expect q, k, v as [B, L, H, D]. Return in the same layout [B, L, H, D].
    Uses FlashAttention if available+enabled, otherwise SDPA with query-chunking.
    """
    # --- FlashAttention path (keeps [B, L, H, D]) ---
    if WAN_USE_FLASH and _flash_attn_func is not None and q.is_cuda:
        return _flash_attn_func(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            dropout_p=0.0, softmax_scale=softmax_scale, causal=causal
        )

    # --- SDPA fallback: SDPA expects [B, H, L, D] ---
    q_sdpa = q.transpose(1, 2).contiguous()  # [B, H, Lq, D]
    k_sdpa = k.transpose(1, 2).contiguous()  # [B, H, Lk, D]
    v_sdpa = v.transpose(1, 2).contiguous()  # [B, H, Lk, D]

    B, H, Lq, _ = q_sdpa.shape
    CHUNK = int(os.getenv("WAN_SDP_CHUNK", "256"))

    outs = []

    # Prefer mem-efficient kernel if available; otherwise fall back to default context
    use_ctx = False
    try:
        from torch.nn.attention import sdpa_kernel as _sdpa_kernel
        ctx = _sdpa_kernel(use_flash=False, use_mem_efficient=True, use_math=False)
        use_ctx = True
    except Exception:
        pass

    for qs in range(0, Lq, CHUNK):
        qe = min(Lq, qs + CHUNK)
        q_chunk = q_sdpa[:, :, qs:qe, :]
        if use_ctx:
            with ctx:
                try:
                    o = F.scaled_dot_product_attention(
                        q_chunk, k_sdpa, v_sdpa,
                        is_causal=causal, dropout_p=0.0, scale=softmax_scale
                    )
                except TypeError:
                    o = F.scaled_dot_product_attention(
                        q_chunk, k_sdpa, v_sdpa,
                        is_causal=causal, dropout_p=0.0
                    )
        else:
            o = F.scaled_dot_product_attention(
                q_chunk, k_sdpa, v_sdpa,
                is_causal=causal, dropout_p=0.0
            )
        outs.append(o)
        del o, q_chunk
        torch.cuda.empty_cache()

    out = torch.cat(outs, dim=2)                 # [B, H, Lq, D]
    return out.transpose(1, 2).contiguous()      # -> [B, Lq, H, D]


class CausalConv1d(nn.Module):

    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode="replicate", **kwargs):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size - 1, 0)  # T
        self.time_causal_padding = padding

        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)



class FaceEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, dtype=None, device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.num_heads = num_heads
        self.conv1_local = CausalConv1d(in_dim, 1024 * num_heads, 3, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 8, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(1024, 1024, 3, stride=2)
        self.conv3 = CausalConv1d(1024, 1024, 3, stride=2)

        self.out_proj = nn.Linear(1024, hidden_dim)
        self.norm1 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.norm2 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.norm3 = nn.LayerNorm(1024, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        
        x = rearrange(x, "b t c -> b c t")
        b, c, t = x.shape

        x = self.conv1_local(x)
        x = rearrange(x, "b (n c) t -> (b n) t c", n=self.num_heads)
        
        x = self.norm1(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv2(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm2(x)
        x = self.act(x)
        x = rearrange(x, "b t c -> b c t")
        x = self.conv3(x)
        x = rearrange(x, "b c t -> b t c")
        x = self.norm3(x)
        x = self.act(x)
        x = self.out_proj(x)
        x = rearrange(x, "(b n) t c -> b t n c", b=b)
        padding = self.padding_tokens.repeat(b, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        return x_local



class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        device=None,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, **factory_kwargs))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        if hasattr(self, "weight"):
            output = output * self.weight
        return output


def get_norm_layer(norm_layer):
    """
    Get the normalization layer.

    Args:
        norm_layer (str): The type of normalization layer.

    Returns:
        norm_layer (nn.Module): The normalization layer.
    """
    if norm_layer == "layer":
        return nn.LayerNorm
    elif norm_layer == "rms":
        return RMSNorm
    else:
        raise NotImplementedError(f"Norm layer {norm_layer} is not implemented")


class FaceAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads_num: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        num_adapter_layers: int = 1,
        dtype=None,
        device=None,
    ):

        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.hidden_size = hidden_dim
        self.heads_num = heads_num
        self.fuser_blocks = nn.ModuleList(
            [
                FaceBlock(
                    self.hidden_size,
                    self.heads_num,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(num_adapter_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        motion_embed: torch.Tensor,
        idx: int,
        freqs_cis_q: Tuple[torch.Tensor, torch.Tensor] = None,
        freqs_cis_k: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:

        return self.fuser_blocks[idx](x, motion_embed, freqs_cis_q, freqs_cis_k)



class FaceBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.scale = qk_scale or head_dim**-0.5
       
        self.linear1_kv = nn.Linear(hidden_size, hidden_size * 2, **factory_kwargs)
        self.linear1_q = nn.Linear(hidden_size, hidden_size, **factory_kwargs)

        self.linear2 = nn.Linear(hidden_size, hidden_size, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.pre_norm_feat = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.pre_norm_motion = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        motion_vec: torch.Tensor,
        motion_mask: Optional[torch.Tensor] = None,
        use_context_parallel=False,
    ) -> torch.Tensor:
        
        B, T, N, C = motion_vec.shape
        T_comp = T

        x_motion = self.pre_norm_motion(motion_vec)
        x_feat = self.pre_norm_feat(x)

        kv = self.linear1_kv(x_motion)
        q = self.linear1_q(x_feat)

        k, v = rearrange(kv, "B L N (K H D) -> K B L N H D", K=2, H=self.heads_num)
        q = rearrange(q, "B S (H D) -> B S H D", H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        k = rearrange(k, "B L N H D -> (B L) N H D")  
        v = rearrange(v, "B L N H D -> (B L) N H D") 

        if use_context_parallel:
            q = gather_forward(q, dim=1)

        q = rearrange(q, "B (L S) H D -> (B L) S H D", L=T_comp)  
        # Compute attention.
        attn = attention(
            q,
            k,
            v,
            max_seqlen_q=q.shape[1],
            batch_size=q.shape[0],
        )

        attn = rearrange(attn, "(B L) S C -> B (L S) C", L=T_comp)
        if use_context_parallel:
            attn = torch.chunk(attn, get_world_size(), dim=1)[get_rank()]

        output = self.linear2(attn)

        if motion_mask is not None:
            output = output * rearrange(motion_mask, "B T H W -> B (T H W)").unsqueeze(-1)

        return output