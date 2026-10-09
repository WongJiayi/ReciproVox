"""
rope_kernel.py — ViT 的 RoPE 内核（接口不变），按 Qwen2-VL 的多模态 3D RoPE 思路实现，
支持四种模式：'mrope' | 'vanilla' | 'tad' | 'videorope'（含时间缩放与空间合并交错）。

- SimpleRotaryEmbedding
- _split_hwt_dims
- _build_thw_indices
- _pack_cos_sin
- compute_cos_sin(Dx, Dy, Dz, head_dim, theta, mode, t_scale, device)
- prepare_vision_rope(encoder, images, patch_size, head_dim, theta, mode, t_scale, device)

与上文 Qwen2-VL 行为对齐要点：
1) mrope：标准 3D 网格索引 (t,h,w) 分段生成 cos/sin 并拼接；
2) vanilla：将 1D 展平索引复制到三轴；
3) tad：time-only（t 复制到 h,w）与 vanilla 的索引**相加**，再生成 cos/sin；
4) videorope：t 轴索引按 t_scale 放大；h,w 以中心为 0 后相对 t 平移（"body-diagonal"），并在通道维对 h 与 w 的频率**交错合并**，再把 t 段放到最后（对应上文的 apply_m_modify_* 行为）。
"""

import torch
import torch.nn as nn
from typing import Tuple

# ---- 基础 ----
class SimpleRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0, dtype=torch.float32):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def sinusoid(self, positions: torch.Tensor):
        """返回 cos, sin；自动适配 positions 的 device。"""
        local_inv = self.inv_freq.to(positions.device)
        freqs = torch.outer(positions.to(local_inv.dtype), local_inv)   # (P, dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)                         # (P, dim)
        return emb.cos(), emb.sin()

def set_shared_axial_freqs(encoder: nn.Module, freqs_nd: torch.Tensor) -> None:
    for layer in encoder.layers:
        layer.self_attn.set_freqs(freqs_nd)


def _split_hwt_dims(head_dim: int) -> Tuple[int, int, int]:
    """按 h:w:t = 3:3:2 分配，确保偶数并和为 head_dim。"""
    assert head_dim % 2 == 0, "head_dim 必须为偶数用于 RoPE"
    ratio = torch.tensor([3.0, 3.0, 2.0])
    raw = (ratio / ratio.sum() * head_dim).round().int()
    # 变成偶数
    raw = raw - (raw % 2)
    raw = torch.clamp(raw, min=2)
    diff = int(head_dim - int(raw.sum()))
    raw[-1] += diff  # 把余数给 t 段
    if (raw[-1] % 2) == 1:
        raw[-1] += 1
        raw[0] -= 1
    return int(raw[0]), int(raw[1]), int(raw[2])  # (h, w, t)


def _build_thw_indices(Dx, Dy, Dz, device):
    # t-major: t -> h -> w
    t = torch.arange(Dz, device=device).view(Dz, 1, 1).expand(Dz, Dy, Dx)
    h = torch.arange(Dy, device=device).view(1, Dy, 1).expand(Dz, Dy, Dx)
    w = torch.arange(Dx, device=device).view(1, 1, Dx).expand(Dz, Dy, Dx)
    return torch.stack([t.flatten(), h.flatten(), w.flatten()], dim=0)  # (3, N) = [t,h,w]


def _pack_cos_sin(cos: torch.Tensor, sin: torch.Tensor):
    # (N, d) -> (N, 2d)
    return torch.cat([cos, sin], dim=-1)


def _interleave_lastdim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """按最后一维交错合并 a,b（形状: (N, Da)/(N, Db)）。"""
    N, Da = a.shape
    _, Db = b.shape
    m = min(Da, Db)
    if m > 0:
        stacked = torch.stack([a[:, :m], b[:, :m]], dim=-1).reshape(N, -1)
    else:
        stacked = a.new_empty(N, 0)
    tails = []
    if Da > m:
        tails.append(a[:, m:])
    if Db > m:
        tails.append(b[:, m:])
    return torch.cat([stacked] + tails, dim=-1) if tails else stacked


# ---- 各模式的 cos/sin 生成 ----
@torch.no_grad()
def compute_cos_sin(
    Dx: int, Dy: int, Dz: int, head_dim: int, theta: float,
    mode: str, t_scale: float, device
):
    """
    返回 packed: (N, 2*head_dim)，N=Dx*Dy*Dz
    mode: 'vanilla' | 'tad' | 'mrope' | 'videorope'
    """
    assert head_dim % 2 == 0

    # 先按 h:w:t=3:3:2 拆段
    dh, dw, dt = _split_hwt_dims(head_dim)

    # 索引 (3,N)
    pos3 = _build_thw_indices(Dx, Dy, Dz, device=device)  # [t,h,w]
    N = pos3.shape[1]

    # 三段的 RoPE 嵌入器
    emb_t = SimpleRotaryEmbedding(dt, theta).to(device)
    emb_h = SimpleRotaryEmbedding(dh, theta).to(device)
    emb_w = SimpleRotaryEmbedding(dw, theta).to(device)

    # 构造三轴的“位置索引”
    if mode == "mrope":
        pt = pos3[0].float()
        ph = pos3[1].float()
        pw = pos3[2].float()

        ct, st = emb_t.sinusoid(pt)
        ch, sh = emb_h.sinusoid(ph)
        cw, sw = emb_w.sinusoid(pw)

        cos_full = torch.cat([ct, ch, cw], dim=-1)
        sin_full = torch.cat([st, sh, sw], dim=-1)

    elif mode == "vanilla":
        flat = torch.arange(N, device=device, dtype=torch.float)
        pt = ph = pw = flat

        ct, st = emb_t.sinusoid(pt)
        ch, sh = emb_h.sinusoid(ph)
        cw, sw = emb_w.sinusoid(pw)

        cos_full = torch.cat([ct, ch, cw], dim=-1)
        sin_full = torch.cat([st, sh, sw], dim=-1)

    elif mode == "tad":
        # time-only 与 vanilla 的索引相加（与上文 Qwen2-VL 的 position_ids 相加一致）
        base_t = pos3[0].float()
        base_v = torch.arange(N, device=device, dtype=torch.float)
        idx = base_t + base_v  # t + flat，三轴相同
        pt = ph = pw = idx

        ct, st = emb_t.sinusoid(pt)
        ch, sh = emb_h.sinusoid(ph)
        cw, sw = emb_w.sinusoid(pw)

        cos_full = torch.cat([ct, ch, cw], dim=-1)
        sin_full = torch.cat([st, sh, sw], dim=-1)

    elif mode == "videorope":
        # 1) t 放大；2) h,w 中心化并相对 t 平移；3) 频率层面合并交错(h↔w)，并把 t 段放到最后
        pt = pos3[0].float() * float(t_scale)
        # 中心化：以网格中心为 0
        h_center = pos3[1].float() - (Dy - 1) / 2.0
        w_center = pos3[2].float() - (Dx - 1) / 2.0
        ph = h_center + pt
        pw = w_center + pt

        ct, st = emb_t.sinusoid(pt)
        ch, sh = emb_h.sinusoid(ph)
        cw, sw = emb_w.sinusoid(pw)

        # 交错合并 h 与 w，然后把 t 放到最后（对应 apply_m_modify_* 的通道次序）
        hw_cos = _interleave_lastdim(ch, cw)
        hw_sin = _interleave_lastdim(sh, sw)

        cos_full = torch.cat([hw_cos, ct], dim=-1)
        sin_full = torch.cat([hw_sin, st], dim=-1)

    else:
        raise ValueError(f"unsupported rope mode: {mode}")

    return _pack_cos_sin(cos_full, sin_full)  # (N, 2*head_dim)


# ---- 对接 encoder 的一次性入口（每个 batch 调用）----
@torch.no_grad()
def prepare_vision_rope(
    encoder,
    images: torch.Tensor,
    patch_size, head_dim: int,
    theta: float, mode: str, t_scale: float,
    device
):
    # 计算 (Dz,Dy,Dx)
    if isinstance(patch_size, int):
        p_d = p_h = p_w = patch_size
    else:
        p_d, p_h, p_w = patch_size
    if images.dim() == 5:
        _, _, D, H, W = images.shape
        Dz = max(1, D // p_d); Dy = max(1, H // p_h); Dx = max(1, W // p_w)
    else:
        _, _, H, W = images.shape
        Dz = 1; Dy = max(1, H // p_h); Dx = max(1, W // p_w)

    packed = compute_cos_sin(Dx, Dy, Dz, head_dim, theta, mode, t_scale, device)
    set_shared_axial_freqs(encoder, packed)  # 复用 setter，把 (N, 2*head_dim) 写进 encoder
