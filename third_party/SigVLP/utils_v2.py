import os
import re
from typing import Iterable, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from muon import MuonWithAuxAdam
from .vision import TextEmbeddingsNoAbsPos


# -------------------------
# Generic helpers
# -------------------------
def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def get_latest_checkpoint(ckpt_dir: str) -> Tuple[Optional[str], int]:
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
        return None, 0

    pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt") and "step" in f]
    if not pt_files:
        return None, 0

    def extract_step(f):
        m = re.search(r"step(\d+)", f)
        return int(m.group(1)) if m else 0

    pt_files.sort(key=extract_step, reverse=True)
    latest = pt_files[0]
    return os.path.join(ckpt_dir, latest), extract_step(latest)


def set_seed(seed=42):
    import random
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


def interpolate_1d(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """
    x: (B, D) -> (B, target_dim) via 1D linear interpolate
    """
    return F.interpolate(x.unsqueeze(1), size=target_dim, mode="linear", align_corners=False).squeeze(1)


# -------------------------
# Vision resize & rebuild (SigLIP v1 shell)
# -------------------------
def _size_config(vit_size: str):
    size_map = {
        "small": {"layers": 12, "hidden": 396,  "heads": 6},
        "base":  {"layers": 12, "hidden": 792,  "heads": 12},
        "large": {"layers": 24, "hidden": 1056, "heads": 16},
    }
    assert vit_size in size_map, f"vit_size must be one of {list(size_map.keys())}"
    return size_map[vit_size]


def resize_vit_and_rebuild_vision_v1(model, *, vit_size: str, patch_size, use_sdpa: bool = True,
                                     mlp_ratio: float = 4.0, image_size: int = 256, in_channels: int = 1):
    """
    只改 vision_config 数字 & 重新实例化 vision_model（v1 外壳），不触碰 text。
    """
    cfg = _size_config(vit_size)

    vconf = model.config.vision_config
    tconf = model.config.text_config

    vconf.num_hidden_layers   = int(cfg["layers"])
    vconf.hidden_size         = int(cfg["hidden"])
    vconf.num_attention_heads = int(cfg["heads"])
    vconf.intermediate_size   = int(vconf.hidden_size * mlp_ratio)
    vconf.image_size          = image_size
    vconf.patch_size          = patch_size
    vconf.num_channels        = in_channels
    setattr(vconf, "_attn_implementation", "sdpa" if use_sdpa else "eager")

    head_dim = vconf.hidden_size // vconf.num_attention_heads
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"

    # 保留 text 侧的投影维度（不要强行改）
    if hasattr(tconf, "projection_size"): tconf.projection_size = getattr(tconf, "projection_size")
    if hasattr(tconf, "projection_dim"):  tconf.projection_dim  = getattr(tconf, "projection_dim")

    # 重新构建 vision（SigLIP v1）
    VisionCls = type(model.vision_model)
    model.vision_model = VisionCls(vconf)

    return {
        "layers": vconf.num_hidden_layers,
        "hidden": vconf.hidden_size,
        "heads": vconf.num_attention_heads,
        "head_dim": head_dim,
        "mlp": vconf.intermediate_size,
        "attn_impl": getattr(vconf, "_attn_implementation", "eager"),
        "image_size": vconf.image_size,
        "patch_size": vconf.patch_size,
        "in_channels": vconf.num_channels,
    }


# -------------------------
# 3D Embeddings
# -------------------------
def inject_3d_embeddings(model, *, in_channels: int, embed_dim: int, patch_size, pos_embed_type: str):
    # 直接复用你在 vision.py 里的实现
    from .vision import SiglipVisionEmbeddings3D
    real = unwrap_model(model)
    real.vision_model.embeddings = SiglipVisionEmbeddings3D(
        in_channels=in_channels,
        embed_dim=embed_dim,
        patch_size=patch_size,
        pos_embed_type=pos_embed_type,
    )


# -------------------------
# Projection alignment (vision->text)
# -------------------------
def _get_text_proj_dim(model) -> int:
    # HF SigLIP(v1) 通常有 text_projection 参数 (hidden_text -> projection_dim)
    if hasattr(model, "text_projection") and isinstance(model.text_projection, torch.nn.Parameter):
        return int(model.text_projection.shape[1])
    # 配置兜底
    d = getattr(model.config, "projection_dim", None)
    if d is None and hasattr(model, "config") and hasattr(model.config, "text_config"):
        d = getattr(model.config.text_config, "projection_size", None) or getattr(
            model.config.text_config, "projection_dim", None
        )
    return int(d) if d is not None else 768


def _set_all_visual_proj_aliases(model, param: torch.nn.Parameter):
    # 兼容不同命名
    setattr(model, "visual_projection", param)
    setattr(model, "vision_projection", param)
    setattr(model, "image_projection", param)


def align_siglip_projection_to_text(model, target_dim: int = None):
    """
    把视觉侧的 projection（如果是顶层 Parameter）对齐到文本侧投影维度。
    - 若已存在且 shape 匹配，则复用；
    - 否则新建 (hidden_vis, target_dim) 并尽量拷贝原有列。
    注意：不强制修改 text_projection，保持预训练语义空间稳定。
    """
    real = unwrap_model(model)

    if target_dim is None:
        target_dim = _get_text_proj_dim(real)

    vconf = getattr(real.config, "vision_config", None)
    if vconf is None:
        return  # 不是 SigLIP 结构，跳过
    vis_in = int(vconf.hidden_size)

    device = next(real.parameters()).device
    dtype = next(real.parameters()).dtype

    # 已存在且匹配就直接绑定别名并返回
    for name in ("visual_projection", "vision_projection", "image_projection"):
        w = getattr(real, name, None)
        if isinstance(w, torch.nn.Parameter) and w.ndim == 2 and w.shape == (vis_in, target_dim):
            _set_all_visual_proj_aliases(real, w)
            if hasattr(real.config, "projection_dim"):
                real.config.projection_dim = target_dim
            if hasattr(real.config, "vision_config"):
                real.config.vision_config.projection_dim = target_dim
            if hasattr(real.config, "text_config"):
                if hasattr(real.config.text_config, "projection_size"):
                    real.config.text_config.projection_size = target_dim
                if hasattr(real.config.text_config, "projection_dim"):
                    real.config.text_config.projection_dim = target_dim
            return

    # 新建并尽量拷贝
    new_w = torch.empty(vis_in, target_dim, dtype=dtype, device=device)
    init_factor = getattr(real.config, "initializer_factor", 1.0)
    nn.init.normal_(new_w, std=0.02 * init_factor)

    for name in ("visual_projection", "vision_projection", "image_projection"):
        w = getattr(real, name, None)
        if isinstance(w, torch.nn.Parameter) and w.ndim == 2 and w.shape[0] == vis_in:
            with torch.no_grad():
                c = min(w.shape[1], target_dim)
                new_w[:, :c] = w.data.to(dtype=dtype, device=device)[:, :c]
                if c < target_dim:
                    nn.init.normal_(new_w[:, c:], std=0.02 * init_factor)
            break

    new_param = nn.Parameter(new_w, requires_grad=True)
    _set_all_visual_proj_aliases(real, new_param)

    if hasattr(real.config, "projection_dim"):
        real.config.projection_dim = target_dim
    if hasattr(real.config, "vision_config"):
        real.config.vision_config.projection_dim = target_dim
    if hasattr(real.config, "text_config"):
        if hasattr(real.config.text_config, "projection_size"):
            real.config.text_config.projection_size = target_dim
        if hasattr(real.config.text_config, "projection_dim"):
            real.config.text_config.projection_dim = target_dim


# -------------------------
# Text RoPE init (once)
# -------------------------
def init_text_rope_once(model, base: float = 10000.0, max_len: Optional[int] = None) -> bool:
    """
    1) 把 text embeddings 换成无 abs pos 的版本（vision.TextEmbeddingsNoAbsPos）
    2) 用通用 RoPE 包装 text encoder 的 self_attn（models.patch_siglip_rope_text_generic）
    3) 按 max_len 预生成 1D RoPE（packed [cos|sin]）并设置为共享频率
    """
    real = unwrap_model(model)

    # 1) swap embeddings
    from vision import TextEmbeddingsNoAbsPos
    txt = getattr(real, "text_model", None)
    if txt is None:
        raise ValueError("model.text_model 不存在。")
    if not isinstance(txt.embeddings, TextEmbeddingsNoAbsPos):
        txt.embeddings = TextEmbeddingsNoAbsPos(txt.embeddings)

    # 2) patch attention
    from models.rope_generic import patch_encoder_with_rope as patch_text_with_rope
    from models.rope_generic import set_shared_axial_freqs as text_set_shared_freqs
    from models.rope_generic import build_rope_packed as text_build_packed

    enc = txt.encoder
    # 已经是 RopeSelfAttention 则不重复
    patched = False
    for lyr in enc.layers:
        if not hasattr(lyr.self_attn, "set_freqs"):
            patched = True
            break
    if patched:
        patch_text_with_rope(enc)

    # 3) set shared 1D freqs
    if max_len is None:
        # 取 config 中可用长度
        max_len = getattr(real.config.text_config, "max_position_embeddings", 16)
    head_dim = getattr(real.config.text_config, "hidden_size", 768) // getattr(
        real.config.text_config, "num_attention_heads", 12
    )
    assert head_dim % 2 == 0, f"text head_dim must be even for RoPE, got {head_dim}"

    freqs = text_build_packed(L=max_len, D=head_dim, base=base, device=next(real.parameters()).device,
                              dtype=next(real.parameters()).dtype)  # (L, D) packed
    text_set_shared_freqs(enc, freqs)  # 所有层共享
    return True

def configure_text_pos_mode(model, *, mode: str, rope_base: float = 10000.0, max_seq_len: int = 16) -> dict:
    """
    选择文本侧位置编码：
      - mode='abs'  : 使用默认（绝对位置），不打 RoPE 补丁
      - mode='rope' : 替换为无绝对位置 + 打 RoPE + 设置频率
    返回：{'mode':..., 'base':..., 'max_len':...} 方便日志
    """
    real = unwrap_model(model)

    mode = str(mode).lower()
    if mode not in ("abs", "rope"):
        raise ValueError(f"text_pos_mode must be 'abs' or 'rope', got {mode}")

    if mode == "abs":
        # 重建 Text 后默认就是 abs；如果之前被换成 NoAbs/贴了 RoPE，最安全的方式是“先重建再 abs”
        # 这里不做任何额外动作（重建由上层调用保证）
        return {"mode": "abs"}

    # == rope ==
    # 幂等：多次调用也不会重复 patch/重复下发频率
    init_text_rope_once(real, base=rope_base, max_len=max_seq_len)
    return {"mode": "rope", "base": float(rope_base), "max_len": int(max_seq_len)}


# -------------------------
# Reshape: resize Text encoder
# -------------------------
def _size_config_text(text_size: str):
    # 与 vision 对齐：small=396/6/12, base=792/12/12, large=1056/16/24
    return _size_config(text_size)  # 直接复用你已有的 _size_config

def resize_text_and_rebuild_v1(model, *, text_size: str,
                               use_sdpa: bool = True,
                               mlp_ratio: float = 4.0,
                               vocab_size: int = None,
                               max_position_embeddings: int = 16):
    """
    只改 text_config 数字 & 重新实例化 text_model（SigLIP v1 外壳），不触碰 vision。
    注意：会根据 text_size 设置 hidden/heads/layers，并把 MLP 设为 hidden*mlp_ratio。
    """
    real = unwrap_model(model)
    cfg = _size_config_text(text_size)

    tconf = real.config.text_config
    tconf.num_hidden_layers   = int(cfg["layers"])
    tconf.hidden_size         = int(cfg["hidden"])
    tconf.num_attention_heads = int(cfg["heads"])
    tconf.intermediate_size   = int(tconf.hidden_size * mlp_ratio)
    tconf.max_position_embeddings = int(max_position_embeddings)
    setattr(tconf, "_attn_implementation", "sdpa" if use_sdpa else "eager")

    if hasattr(tconf, "projection_size"):
        tconf.projection_size = tconf.hidden_size
    if hasattr(tconf, "projection_dim"):
        tconf.projection_dim = tconf.hidden_size

    if vocab_size is not None:
        tconf.vocab_size = int(vocab_size)

    # 重新构建 text（SigLIP v1）
    TextCls = type(real.text_model)
    real.text_model = TextCls(tconf)

    head_dim = tconf.hidden_size // tconf.num_attention_heads
    assert head_dim % 2 == 0, f"[Text] head_dim must be even for RoPE, got {head_dim}"

    return {
        "layers": tconf.num_hidden_layers,
        "hidden": tconf.hidden_size,
        "heads": tconf.num_attention_heads,
        "head_dim": head_dim,
        "mlp": tconf.intermediate_size,
        "attn_impl": getattr(tconf, "_attn_implementation", "eager"),
        "vocab_size": getattr(tconf, "vocab_size", None),
        "max_pos": tconf.max_position_embeddings,
    }

# -------------------------
# Assert: no stray 768
# -------------------------
def assert_no_dim_768_left(model, expected_hidden: int):
    """
    检查 vision/text 里是否还有特定的 768 残留（Linear in/out 或 LayerNorm 维度等）。
    """
    real = unwrap_model(model)
    offenders = []

    def _check_module(prefix, m: nn.Module):
        nonlocal offenders
        if isinstance(m, nn.Linear):
            if m.in_features == 768 and m.in_features != expected_hidden:
                offenders.append(f"{prefix}.in={m.in_features} (Linear)")
            if m.out_features == 768 and m.out_features != expected_hidden:
                offenders.append(f"{prefix}.out={m.out_features} (Linear)")
        elif isinstance(m, nn.LayerNorm):
            if tuple(m.normalized_shape) == (768,) and 768 != expected_hidden:
                offenders.append(f"{prefix}.norm=768 (LayerNorm)")

    for name, mod in real.named_modules():
        _check_module(name, mod)

    if offenders:
        msg = "⚠️ Found 768-shaped leftovers in modules that should use hidden_size " \
              f"{expected_hidden}:\n  - " + "\n  - ".join(offenders)
        raise AssertionError(msg)

"""
# -------------------------
# Metric (val) + PR 曲线到 W&B
# -------------------------
from contextlib import nullcontext

@torch.no_grad()
def compute_metrics(
    model,
    val_dataloader,
    tokenizer,
    device,
    max_txt_len: int = 16,
    log_to_wandb: bool = True,
    wandb_prefix: str = "eval",
):
    # Retrieval evaluation (1:1 image<->text) + binary metrics from full similarity matrix:
    #  - i2t, t2i: Recall@K, MeanRank
    #  - PR curve + AP(PR-AUC), ROC curve + ROC-AUC
    #  - Best-F1 (扫描阈值得到), 以及该阈值下的 Precision/Recall/Acc
    #  - Top-1 混淆矩阵（i->t 与 t->i），观察对角线

    try:
        real = unwrap_model(model)
    except NameError:
        real = model

    real.eval()

    all_img, all_txt = [], []
    K = 10
    for b, batch in enumerate(val_dataloader):
        if b >= K: break
        images = batch["pixel_values"].to(device)
        texts  = batch["text"]

        # tokenize text
        text_inputs = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=max_txt_len, return_tensors="pt",
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        # text embed
        if hasattr(real, "get_text_features"):
            t = real.get_text_features(**text_inputs)
        else:
            to = real.text_model(**text_inputs, return_dict=True)
            if getattr(to, "pooler_output", None) is not None:
                t = to.pooler_output
            else:
                eos_idx = text_inputs["attention_mask"].sum(dim=1) - 1
                t = to.last_hidden_state[torch.arange(eos_idx.size(0), device=eos_idx.device), eos_idx]
            if hasattr(real, "text_projection") and isinstance(getattr(real, "text_projection", None), torch.nn.Parameter):
                t = t @ real.text_projection
        t = F.normalize(t, dim=-1)

        # image embed
        vo = real.vision_model(pixel_values=images, return_dict=True)
        v = getattr(vo, "pooler_output", vo)
        if hasattr(real, "visual_projection") and isinstance(getattr(real, "visual_projection", None), torch.nn.Parameter):
            v = v @ real.visual_projection
        v = F.normalize(v, dim=-1)

        all_txt.append(t)
        all_img.append(v)

    text_mat  = torch.cat(all_txt, dim=0)    # (N, D)
    image_mat = torch.cat(all_img, dim=0)    # (N, D)
    assert text_mat.size(0) == image_mat.size(0), "Val set must be 1:1 image↔text."
    N = text_mat.size(0)

    # similarity (I x T)
    sims = image_mat @ text_mat.t()          # (N, N)
    if hasattr(real, "logit_scale"):
        sims = sims * real.logit_scale.exp()
    if hasattr(real, "logit_bias"):
        sims = sims + real.logit_bias

    # ---------- Retrieval metrics ----------
    def recall_at_k(sim, k, dim=1):
        ranks = torch.argsort(sim, dim=dim, descending=True)
        idx   = torch.arange(sim.size(0), device=sim.device)
        if dim == 1:  # i->t
            topk = ranks[:, :k]
            hits = (topk == idx.unsqueeze(1)).any(dim=1).float().mean().item()
            pos  = (ranks == idx.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
        else:         # t->i
            topk = ranks[:k, :].t()
            hits = (topk == idx.unsqueeze(1)).any(dim=1).float().mean().item()
            pos  = (ranks.t() == idx.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
        mean_rank = (pos.float().mean().item() + 1.0)
        return hits, mean_rank

    diag_mean = torch.diag(sims).mean().item()
    r1_i2t, mr_i2t = recall_at_k(sims,   1, dim=1)
    r5_i2t, _      = recall_at_k(sims,   5, dim=1)
    r10_i2t, _     = recall_at_k(sims,  10, dim=1)

    r1_t2i, mr_t2i = recall_at_k(sims.t(), 1, dim=1)
    r5_t2i, _      = recall_at_k(sims.t(), 5, dim=1)
    r10_t2i, _     = recall_at_k(sims.t(),10, dim=1)

    # ---------- Binary metrics from scores ----------
    # labels: diag=1, off-diag=0
    y_true = torch.eye(N, device=sims.device, dtype=torch.int8).flatten().cpu().numpy()
    y_score = sims.flatten().detach().cpu().numpy()

    # 可能非常大（N^2），给个软保护（> 9e6 个样本时跳过曲线绘图以省显存/时间）
    do_curves = (y_true.size <= 9_000_000)

    pr_auc = roc_auc = None
    best_f1 = best_p = best_r = best_acc = None
    thr_at_best_f1 = None
    pr_curve_fig = roc_curve_fig = None

    try:
        from sklearn.metrics import (
            precision_recall_curve, average_precision_score,
            roc_curve, roc_auc_score, confusion_matrix
        )
        if do_curves:
            # PR 曲线 & AP (PR-AUC)
            prec, rec, thr = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            pr_auc = float(ap)

            # ROC 曲线 & AUC
            fpr, tpr, thr_roc = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            roc_auc = float(auc)

            # 扫描阈值取 F1 最大（用 PR 曲线上的阈值）
            eps = 1e-12
            f1s = 2 * prec * rec / (prec + rec + eps)
            best_idx = int(f1s.argmax())
            best_f1 = float(f1s[best_idx])
            thr_at_best_f1 = float(thr[best_idx-1]) if best_idx > 0 and best_idx-1 < len(thr) else float(y_score.mean())

            # 用最佳阈值计算 Precision/Recall/Accuracy
            y_pred = (y_score >= thr_at_best_f1).astype(np.int32)
            tp = int((y_pred * y_true).sum())
            fp = int(y_pred.sum() - tp)
            fn = int(y_true.sum() - tp)
            tn = int(y_true.size - tp - fp - fn)
            best_p = float(tp / max(1, tp + fp))
            best_r = float(tp / max(1, tp + fn))
            best_acc = float((tp + tn) / max(1, y_true.size))

            # 画 PR / ROC 曲线
            if log_to_wandb:
                import matplotlib.pyplot as plt, wandb
                fig1, ax1 = plt.subplots(figsize=(5,4), dpi=140)
                ax1.plot(rec, prec, label=f"AP={ap:.4f}")
                ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision"); ax1.set_title("Precision-Recall")
                ax1.legend(loc="lower left")
                pr_curve_fig = fig1

                fig2, ax2 = plt.subplots(figsize=(5,4), dpi=140)
                ax2.plot(fpr, tpr, label=f"AUC={auc:.4f}")
                ax2.plot([0,1],[0,1], linestyle="--", linewidth=1)
                ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.set_title("ROC")
                ax2.legend(loc="lower right")
                roc_curve_fig = fig2
    except Exception as e:
        print(f"[WARN] sklearn metrics unavailable or failed: {e}")

    # ---------- Top-1 Confusion Matrix (i->t and t->i) ----------
    # i->t
    pred_t = sims.argmax(dim=1).cpu().numpy()          # 预测的文本索引
    true_t = np.arange(N)
    # t->i
    pred_i = sims.argmax(dim=0).cpu().numpy()
    true_i = np.arange(N)

    # 构造 NxN 混淆矩阵（可能很大；仅用于可视化）
    def make_conf_mat(pred, true, num_classes):
        # 稀疏 bincount 构造
        idx = pred * num_classes + true
        cm = np.bincount(idx, minlength=num_classes*num_classes).reshape(num_classes, num_classes)
        return cm

    cm_i2t = make_conf_mat(pred_t, true_t, N)
    cm_t2i = make_conf_mat(pred_i, true_i, N)

    # ---------- Collect ----------
    out = {
        "pairs": N,
        "mean_sim_diag": diag_mean,
        # retrieval
        "i2t/R@1": r1_i2t, "i2t/R@5": r5_i2t, "i2t/R@10": r10_i2t, "i2t/MeanRank": mr_i2t,
        "t2i/R@1": r1_t2i, "t2i/R@5": r5_t2i, "t2i/R@10": r10_t2i, "t2i/MeanRank": mr_t2i,
        # binary metrics
        "bin/PR_AUC": pr_auc if pr_auc is not None else float("nan"),
        "bin/ROC_AUC": roc_auc if roc_auc is not None else float("nan"),
        "bin/BestF1": best_f1 if best_f1 is not None else float("nan"),
        "bin/BestPrec": best_p if best_p is not None else float("nan"),
        "bin/BestRec": best_r if best_r is not None else float("nan"),
        "bin/BestAcc": best_acc if best_acc is not None else float("nan"),
        "bin/Thr@BestF1": thr_at_best_f1 if thr_at_best_f1 is not None else float("nan"),
        # top-1 acc（与 R@1 等价）
        "i2t/Top1Acc": float((pred_t == true_t).mean()),
        "t2i/Top1Acc": float((pred_i == true_i).mean()),
    }

    # ---------- W&B logging ----------
    if log_to_wandb:
        try:
            import wandb, matplotlib.pyplot as plt
            # 相似度直方图
            fig, ax = plt.subplots(figsize=(5, 4), dpi=140)
            ax.hist(sims.detach().cpu().flatten().numpy(), bins=50)
            ax.set_title("Similarity distribution (I·T)")
            ax.set_xlabel("sim"); ax.set_ylabel("count")
            if wandb.run is not None:
                wandb.log({f"{wandb_prefix}/sim_hist": wandb.Image(fig)})
            plt.close(fig)

            # 对角线分布
            fig2, ax2 = plt.subplots(figsize=(5, 4), dpi=140)
            ax2.hist(torch.diag(sims).detach().cpu().numpy(), bins=50)
            ax2.set_title("Diagonal similarity (positive pairs)")
            ax2.set_xlabel("sim"); ax2.set_ylabel("count")
            if wandb.run is not None:
                wandb.log({f"{wandb_prefix}/sim_diag_hist": wandb.Image(fig2)})
            plt.close(fig2)

            # PR / ROC 曲线
            if pr_curve_fig is not None and wandb.run is not None:
                wandb.log({f"{wandb_prefix}/PR_curve": wandb.Image(pr_curve_fig)})
                plt.close(pr_curve_fig)
            if roc_curve_fig is not None and wandb.run is not None:
                wandb.log({f"{wandb_prefix}/ROC_curve": wandb.Image(roc_curve_fig)})
                plt.close(roc_curve_fig)

            # Top-1 混淆矩阵（热力图）
            # 注意：N 很大时图会很大，这里只在 N<=256 时画全尺寸；更大时下采样/不画
            if N <= 256:
                import seaborn as sns
                fig3, ax3 = plt.subplots(figsize=(6, 5), dpi=140)
                sns.heatmap(cm_i2t, ax=ax3, cmap="viridis", cbar=False)
                ax3.set_title("Confusion (Image→Text); diagonal=correct")
                ax3.set_xlabel("True text idx"); ax3.set_ylabel("Pred text idx")
                wandb.log({f"{wandb_prefix}/confmat_i2t": wandb.Image(fig3)})
                plt.close(fig3)

                fig4, ax4 = plt.subplots(figsize=(6, 5), dpi=140)
                sns.heatmap(cm_t2i, ax=ax4, cmap="viridis", cbar=False)
                ax4.set_title("Confusion (Text→Image); diagonal=correct")
                ax4.set_xlabel("True image idx"); ax4.set_ylabel("Pred image idx")
                wandb.log({f"{wandb_prefix}/confmat_t2i": wandb.Image(fig4)})
                plt.close(fig4)

            # 最后把所有标量一起打点
            if wandb.run is not None:
                wandb.log({f"{wandb_prefix}/{k}": v for k, v in out.items()})
        except Exception as e:
            print(f"[WARN] W&B logging failed: {e}")

    return out
"""

@torch.no_grad()
def compute_metrics(
    model,                  
    dataloader,
    tokenizer,
    device,
    max_txt_len: int = 16,
    log_to_wandb: bool = True,
    wandb_prefix: str = "eval",
    make_heatmap: bool = True, 
    step=None  
):
    """
    只评估 dataloader 的第一个 batch（1:1 图文配对）：
      - 直接使用 SigLIP2 的输出 logits_per_image / logits_per_text
      - 计算 i->t / t->i 的 R@K、MeanRank、Top1Acc
      - 可选绘制行softmax百分比热力图（Image→Text）
    返回:
      dict 指标；并在需要时附带 'logits_per_image' / 'logits_per_text'
    """
    model.eval()
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        return {"error": "empty dataloader"}

    images = batch["pixel_values"].to(device, non_blocking=True)
    texts  = batch["text"]

    ti = tokenizer(
        texts, padding="max_length", truncation=True,
        max_length=max_txt_len, return_tensors="pt"
    )
    ti = {k: v.to(device, non_blocking=True) for k, v in ti.items()}

    # 前向：一次拿全（logits & embeds）
    outputs = model(
        input_ids=ti["input_ids"],
        pixel_values=images,
        return_dict=True
    )

    sims_i2t = outputs.logits_per_image  # (B, B) image→text
    sims_t2i = outputs.logits_per_text   # (B, B) text→image（通常等于 sims_i2t.t()）

    B = sims_i2t.size(0)
    idx = torch.arange(B, device=sims_i2t.device)

    def recall_at_k(sim, k):
        k = min(k, sim.size(1))
        ranks = torch.argsort(sim, dim=1, descending=True)   # 每行 rank
        topk  = ranks[:, :k]
        hits  = (topk == idx.unsqueeze(1)).any(dim=1).float().mean().item()
        pos   = (ranks == idx.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
        mean_rank = (pos.float().mean().item() + 1.0)
        return hits, mean_rank

    # i->t
    r1_i2t, mr_i2t = recall_at_k(sims_i2t, 1)
    r5_i2t,  _     = recall_at_k(sims_i2t, 5)
    r10_i2t, _     = recall_at_k(sims_i2t, 10)
    # t->i
    r1_t2i, mr_t2i = recall_at_k(sims_t2i, 1)
    r5_t2i,  _     = recall_at_k(sims_t2i, 5)
    r10_t2i, _     = recall_at_k(sims_t2i, 10)

    # Top-1（与 R@1 等价，仅直观展示）
    top1_i2t = float((sims_i2t.argmax(dim=1).cpu().numpy() == np.arange(B)).mean())
    top1_t2i = float((sims_t2i.argmax(dim=1).cpu().numpy() == np.arange(B)).mean())

    out = {
        "pairs_in_batch": int(B),
        "i2t/R@1": r1_i2t,  "i2t/R@5": r5_i2t,  "i2t/R@10": r10_i2t,  "i2t/MeanRank": mr_i2t,
        "t2i/R@1": r1_t2i,  "t2i/R@5": r5_t2i,  "t2i/R@10": r10_t2i,  "t2i/MeanRank": mr_t2i,
        "i2t/Top1Acc": top1_i2t,
        "t2i/Top1Acc": top1_t2i,
        #"mean_sim_diag": float(torch.diag(sims_i2t).mean().item()),
        # 如需后续分析可放回 logits（注意：可能较大）
        #"logits_per_image": sims_i2t.detach().cpu(),
        #"logits_per_text":  sims_t2i.detach().cpu(),
    }

    # ===== 可视化：行softmax百分比热力图（Image→Text） =====
    if make_heatmap:
        try:
            import matplotlib.pyplot as plt

            with torch.no_grad():
                mat = sims_i2t.detach().float().cpu().numpy()  # (B,B)
                # 行 softmax
                mat = mat - mat.max(axis=1, keepdims=True)
                prob = np.exp(mat)
                prob /= prob.sum(axis=1, keepdims=True) + 1e-12

            fig, ax = plt.subplots(figsize=(min(12, B*0.3+2), min(12, B*0.3+2)))
            im = ax.imshow(prob, aspect="auto")  # 默认 colormap
            ax.set_title(f"Image→Text softmax % (B={B})")
            ax.set_xlabel("Text index"); ax.set_ylabel("Image index")
            ax.set_xticks(range(B)); ax.set_yticks(range(B))

            # 单元格标注百分比（B=50 仍可读；更大可只标 top-k）
            for i in range(B):
                for j in range(B):
                    p = int(round(prob[i, j] * 100))
                    txt = f"{p}%" if p >= 1 else "0%"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8)

            # 对角线外框
            for i in range(B):
                rect = plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False, linewidth=1.5)
                ax.add_patch(rect)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()

            if log_to_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({f"{wandb_prefix}/batch_heatmap": wandb.Image(fig)}, step=step)
                except Exception as e:
                    print(f"[WARN] wandb log heatmap failed: {e}")
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] heatmap plotting failed: {e}")

    return out


# -------------------------
# Optimizer (Muon + Adam)
# -------------------------
from typing import Iterable, List, Tuple
import torch
from muon import MuonWithAuxAdam

def _split_params_by_dim(params: Iterable[torch.nn.Parameter]) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """ndim>=2 归到权重(走 Muon)，ndim<2 归到 bias/Norm(走 Adam)。"""
    weights, bias_norm = [], []
    for p in params:
        if not p.requires_grad:
            continue
        (weights if p.ndim >= 2 else bias_norm).append(p)
    return weights, bias_norm

def build_optimizer_with_muon(model, lr: float, weight_decay: float):
    real = unwrap_model(model)

    # vision / text 主干
    img_w, img_b = _split_params_by_dim(real.vision_model.parameters())
    txt_w, txt_b = _split_params_by_dim(real.text_model.parameters())

    # 额外顶层参数（一起放到 Adam 组里）
    extra = []
    if hasattr(real, "projection"):
        for p in real.projection.parameters():
            if p.requires_grad:
                extra.append(p)
    if hasattr(real, "visual_projection") and isinstance(real.visual_projection, torch.nn.Parameter):
        if real.visual_projection.requires_grad:
            extra.append(real.visual_projection)
    if hasattr(real, "text_projection") and isinstance(real.text_projection, torch.nn.Parameter):
        if real.text_projection.requires_grad:
            extra.append(real.text_projection)
    if hasattr(real, "logit_scale") and getattr(real.logit_scale, "requires_grad", False):
        extra.append(real.logit_scale)
    if hasattr(real, "logit_bias") and getattr(real.logit_bias, "requires_grad", False):
        extra.append(real.logit_bias)

    # 两组就够了：隐藏权重走 Muon，偏置/Norm/投影/温度走 Adam
    param_groups = [
        dict(params=[*img_w, *txt_w], use_muon=True,  lr=lr, weight_decay=weight_decay),
        dict(params=[*img_b, *txt_b, *extra], use_muon=False, lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay),
    ]
    return MuonWithAuxAdam(param_groups)


def swap_text_to_no_abs_pos(model) -> bool:
    """
    将 text_model.embeddings 替换为不含绝对位置的版本（TextEmbeddingsNoAbsPos）。
    - 兼容 DDP：自动 unwrap
    - 已经替换过则返回 False（不重复替换）
    返回：是否发生了替换（True/False）
    """
    real = unwrap_model(model)

    txt = getattr(real, "text_model", None)
    if txt is None:
        raise ValueError("swap_text_to_no_abs_pos: model.text_model 未找到。")

    emb = getattr(txt, "embeddings", None)
    if emb is None:
        raise ValueError("swap_text_to_no_abs_pos: text_model.embeddings 未找到。")

    if TextEmbeddingsNoAbsPos is None:
        raise ImportError(
            "swap_text_to_no_abs_pos 需要 vision.TextEmbeddingsNoAbsPos。"
            "请确认 vision.py 中已定义并且模块路径可被导入。"
        )

    if isinstance(emb, TextEmbeddingsNoAbsPos):
        # 已经是无绝对位置的版本了
        return False

    txt.embeddings = TextEmbeddingsNoAbsPos(emb)
    return True
