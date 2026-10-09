# --- monkey_patch_siglip_rope.py（可直接粘贴到 train_new.py 顶部或引入） ---
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.siglip.modeling_siglip import SiglipAttention

def enable_text_rope_in_attention(rope_base: float = 10000.0):
    """
    给 SiglipAttention 打补丁，在 forward 里对文本 q/k 应用 RoPE。
    """
    if hasattr(SiglipAttention, "_orig_forward_text"):
        print("[Text RoPE] already patched")
        return
    SiglipAttention._orig_forward_text = SiglipAttention.forward

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def patched_forward(self, hidden_states, *args, **kwargs):
        # 调用原始 forward 拿到 q, k, v
        B, L, C = hidden_states.shape
        # --- 取 qkv ---
        if hasattr(self, "qkv"):  # 有些实现用 qkv projection
            qkv = self.qkv(hidden_states)
            q, k, v = qkv.split(C, dim=-1)
        else:  # HuggingFace SiglipAttention 是分开的
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        num_heads = self.num_heads
        head_dim = q.shape[-1] // num_heads

        def shape(x):  # (B, L, num_heads, head_dim)
            return x.view(B, L, num_heads, head_dim).transpose(1, 2).contiguous()

        q, k, v = map(shape, (q, k, v))

        # --- 动态计算 RoPE ---
        device = q.device
        pos = torch.arange(L, device=device, dtype=torch.float32)          # (L,)
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        freqs = torch.outer(pos, inv_freq)                                # (L, head_dim/2)
        cos = freqs.cos().repeat_interleave(2, dim=-1)[None, None, :, :]  # (1,1,L,head_dim)
        sin = freqs.sin().repeat_interleave(2, dim=-1)[None, None, :, :]  # (1,1,L,head_dim)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # --- 注意力 ---
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_mask = kwargs.get("attention_mask", None)
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v)

        # --- 合并 heads ---
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, num_heads * head_dim)
        if hasattr(self, "proj"):
            out = self.proj(attn_output)
        elif hasattr(self, "out_proj"):
            out = self.out_proj(attn_output)
        else:
            out = attn_output

        return out, None  # 保持原始接口

    SiglipAttention.forward = patched_forward
    print(f"[Text RoPE] Enabled dynamic RoPE in SiglipAttention (base={rope_base})")


def monkey_patch_siglip_rope():

    # ---- 基础工具 ----
    def rotate_half(x):
        # (..., 2*d) -> split into two d parts
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope_to_qk(q, k, packed_freqs, head_dim):
        """
        q, k: (B, H, L, Hd)
        packed_freqs: (L, 2*Hd)  或 (1,1,L,2*Hd) 皆可
        """
        if packed_freqs.dim() == 2:
            # (L, 2*Hd) -> (1,1,L,2*Hd)
            packed = packed_freqs.unsqueeze(0).unsqueeze(0)
        elif packed_freqs.dim() == 4:
            packed = packed_freqs
        else:
            raise ValueError(f"Unexpected freqs shape: {packed_freqs.shape}")

        # cos/sin 拆分并广播到 (B,H,L,Hd)
        cos = packed[..., :head_dim]     # (1,1,L,Hd)
        sin = packed[..., head_dim:]     # (1,1,L,Hd)

        # 旋转
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k

    # ---- 给类加 set_freqs（供 set_shared_axial_freqs 调用）----
    def set_freqs(self, freqs):
        """
        freqs: 期望 (L, 2*head_dim) 或已带 batch/head 维的 (1,1,L,2*head_dim)
        注意：L 要与当前注意力输入的序列长度一致（视觉 patch 数），
        你在 prepare_vision_rope 里已经按 (Dx*Dy*Dz) 生成好了。
        """
        self.rope_freqs = freqs

    # 绑定到类（所有实例共享）
    SiglipAttention.set_freqs = set_freqs

    # ---- 包装 forward：在 q/k 上应用 RoPE ----
    if not hasattr(SiglipAttention, "_orig_forward"):
        SiglipAttention._orig_forward = SiglipAttention.forward  # 保存原 forward

    def patched_forward(self, hidden_states, *args, **kwargs):
        """
        兼容两种形态：
        A) self.qkv 存在：一次性线性 -> 切成 q,k,v
        B) self.q_proj/k_proj/v_proj 分开
        输出继续走原有投影/残差等（尽量保持行为一致）
        """
        # ---- 1. 取到 q,k,v ----
        B, L, C = hidden_states.shape

        if hasattr(self, "qkv"):
            qkv = self.qkv(hidden_states)  # (B, L, 3*D)
            q, k, v = qkv.split(C, dim=-1) if qkv.shape[-1] == 3 * C else torch.chunk(qkv, 3, dim=-1)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # Head 维度信息
        H = getattr(self, "num_heads", None)
        Hd = getattr(self, "head_dim", None)
        if H is None or Hd is None:
            # 部分实现用 embed_dim / num_heads 推出
            embed_dim = q.shape[-1]
            if hasattr(self, "num_heads"):
                H = self.num_heads
                Hd = embed_dim // H
            elif hasattr(self, "head_dim"):
                Hd = self.head_dim
                H = embed_dim // Hd
            else:
                raise AttributeError("Cannot infer num_heads/head_dim from SiglipAttention.")

        # ---- 2. 变形到 (B, H, L, Hd) ----
        def reshape_to_bhlh(x):
            return x.view(B, L, H, Hd).transpose(1, 2).contiguous()  # (B,H,L,Hd)

        q = reshape_to_bhlh(q)
        k = reshape_to_bhlh(k)
        v = reshape_to_bhlh(v)

        # ---- 3. 应用 RoPE（若已设置 freqs）----
        freqs = getattr(self, "rope_freqs", None)
        if freqs is not None:
            # 允许 freqs 在 GPU/CPU 上不一致，自动搬运
            freqs = freqs.to(q.device)
            # 长度对齐（若外面传进来的更长，按 L 截断；若更短，报错）
            if freqs.dim() == 2:
                if freqs.shape[0] < L:
                    raise ValueError(f"rope_freqs length {freqs.shape[0]} < sequence length {L}")
                if freqs.shape[0] > L:
                    freqs = freqs[:L]
            elif freqs.dim() == 4:
                if freqs.shape[-2] < L:
                    raise ValueError(f"rope_freqs length {freqs.shape[-2]} < sequence length {L}")
                if freqs.shape[-2] > L:
                    freqs = freqs[..., :L, :]
            q, k = apply_rope_to_qk(q, k, freqs, Hd)
            #k, q = apply_rope_to_qk(q, k, freqs, Hd)

        # ---- 4. 点积注意力 ----
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (Hd ** 0.5)  # (B,H,L,L)

        # 可选的 attention_mask（适配 kwargs）
        attn_mask = kwargs.get("attention_mask", None)
        if attn_mask is not None:
            # 期望 (B,1,1,L) 或 (B,1,L,L)；最简单以加法方式处理
            attn_scores = attn_scores + attn_mask

        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_probs, v)  # (B,H,L,Hd)

        # ---- 5. 合并头 & 输出投影 ----
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, H * Hd)  # (B,L,C)

        # 兼容不同命名的输出线性层
        if hasattr(self, "proj"):
            out = self.proj(attn_output)
        elif hasattr(self, "out_proj"):
            out = self.out_proj(attn_output)
        elif hasattr(self, "o_proj"):
            out = self.o_proj(attn_output)
        else:
            # 如果原 forward 还做了别的（dropout/residual），退回原 forward
            # 用原 forward 计算，但把我们算好的结果替代其中的中间变量并返回
            return SiglipAttention._orig_forward(self, hidden_states, *args, **kwargs)

        return out

    # 绑定 patched forward
    SiglipAttention.forward = patched_forward
    print("[RoPE] SiglipAttention monkey-patched with set_freqs() and RoPE-enabled forward().")

# --- 使用方式 ---
# 在构建好模型后调用一次（确保 import 了 transformers）：
# model = AutoModel.from_pretrained(...);  # 先有类定义
# monkey_patch_siglip_rope()
# 之后再跑你的 prepare_vision_rope(...)-> set_shared_axial_freqs(...)-> set_freqs() 流程


def _resize_abs_pos_embedding(text_embeddings: nn.Module, text_config, max_seq_len: int, zero_init: bool = False):
    """
    适配 SigLIP 的 text_model.embeddings.position_embedding
    """
    if not hasattr(text_embeddings, "position_embedding"):
        # fallback: 有些实现可能没有显式位置表
        if hasattr(text_config, "max_position_embeddings"):
            text_config.max_position_embeddings = max_seq_len
        print(f"[Text Pos] No position_embedding found. Only updated config.max_position_embeddings={max_seq_len}.")
        return

    old_pe: nn.Embedding = text_embeddings.position_embedding
    old_len, hidden = old_pe.weight.shape

    new_pe = nn.Embedding(max_seq_len, hidden)
    with torch.no_grad():
        if zero_init:
            new_pe.weight.zero_()
        else:
            copy_n = min(old_len, max_seq_len)
            new_pe.weight[:copy_n].copy_(old_pe.weight[:copy_n])
            if max_seq_len > copy_n:
                std = float(old_pe.weight.std().item() if old_len > 0 else 0.02)
                nn.init.normal_(new_pe.weight[copy_n:], mean=0.0, std=std)

    text_embeddings.position_embedding = new_pe

    # 同步修改 config（forward 会检查这个）
    if hasattr(text_config, "max_position_embeddings"):
        text_config.max_position_embeddings = max_seq_len

    print(f"[Text Pos] Resized ABS pos embedding from {old_len} -> {max_seq_len} "
          f"{'(zeroed)' if zero_init else ''}.")

def _patch_text_embeddings_remove_abs(text_embeddings: nn.Module):
    """
    Monkey patch SigLIP 的文本 embeddings：
    - 不再做长度检查
    - 不再加 position_embedding
    - 只返回 token_embedding(input_ids) 或 inputs_embeds
    """
    def forward_no_pos(self, input_ids=None, position_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        return inputs_embeds

    text_embeddings.forward = types.MethodType(forward_no_pos, text_embeddings)

    # 删除原本的 position_embedding 参数，避免 load_state_dict 冲突
    if hasattr(text_embeddings, "position_embedding"):
        delattr(text_embeddings, "position_embedding")

    print("[Text Pos] Patched: removed ABS position addition & length check.")


def configure_text_pos_mode(model, mode: str, rope_base: float = None, max_seq_len: int = 16):
    """
    配置 SigLIP 文本分支的位置编码方式（接口保持不变）。
    - mode='abs' : 保持默认（不改动）
    - mode='rope': 去掉 ABS 位置相加（patch embeddings.forward），RoPE 在 attention 里自行应用
    """
    text_model = getattr(model, "text_model", None)
    if text_model is None:
        raise ValueError("model.text_model not found (is this a SigLIP model?)")

    text_embeddings = getattr(text_model, "embeddings", None)
    text_config = getattr(text_model, "config", None)
    if text_embeddings is None or text_config is None:
        raise ValueError("SigLIP text_model missing embeddings/config")

    if mode == "abs":
        print(f"[Text Pos] ABS mode: keep SigLIP default (position_embedding active). "
              f"max_len stays {getattr(text_config, 'max_position_embeddings', 'unknown')}.")
        return

    if mode == "rope":
        # 1) 去掉 position embedding 的相加 & 长度检查
        _patch_text_embeddings_remove_abs(text_embeddings)

        # 2) 避免长度检查出错：直接给个超大值
        if hasattr(text_config, "max_position_embeddings"):
            text_config.max_position_embeddings = int(1e9)

        print(f"[Text Pos] ROPE mode: disabled ABS pos in embeddings; "
              f"set max_position_embeddings=1e9. Apply RoPE inside attention.")
        return

    raise ValueError(f"Unknown text pos mode: {mode}")
