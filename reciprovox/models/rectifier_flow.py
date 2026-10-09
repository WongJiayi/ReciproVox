"""阶段二 v_θ — rectified flow，起点是 ẑ_mean 而非高斯噪声。

x_t    = (1-t) * ẑ_mean + t * z_GT
target = z_GT - ẑ_mean
L      = ||v_θ(t, x_t, c) - target||²  +  λ_c * L_consistency

条件 c 走主干现成的 cond_image 通路（16 通道，逐像素对齐，
不经过 model_max_length=1 那个失效的 cross-attention）。

两种可训模式（readme §3.4）:
  cond        只放开 x_embedder + final_layer，主干全冻结。最省，先做这个。
  controlnet  再加前 N 层 block 的可训副本 + zero-init 注入。
"""
import copy
import torch
import torch.nn as nn


class Rectifier(nn.Module):
    def __init__(self, backbone, mode="cond", n_control=12, hidden=1024):
        super().__init__()
        assert mode in ("cond", "controlnet")
        self.net, self.mode = backbone, mode
        for p in self.net.parameters():
            p.requires_grad = False                      # 主干默认全冻结

        m = self.net.model
        for p in m.x_embedder.parameters():  p.requires_grad = True
        for p in m.final_layer.parameters(): p.requires_grad = True

        # ---- 第二时间输入 gap = s - t（SplitMeanFlow / MeanFlow 用）----
        # 零初始化输出层：gap=0 时贡献恰好为 0，模型退化成瞬时速度场 v_θ(x,t)，
        # 因此可以从已训好的 Rectifier 无损接续。
        import copy as _c
        self.gap_embedder = _c.deepcopy(m.t_embedder)
        self.gap_proj = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.gap_proj.weight); nn.init.zeros_(self.gap_proj.bias)
        for p_ in self.gap_embedder.parameters(): p_.requires_grad = True
        for p_ in self.gap_proj.parameters():     p_.requires_grad = True

        if mode == "controlnet":
            self.n_control = n_control
            self.ctrl = nn.ModuleList([copy.deepcopy(m.blocks[i]) for i in range(n_control)])
            for p in self.ctrl.parameters(): p.requires_grad = True
            # zero-init 注入：第一步贡献恰好为 0 -> 主干输出与原模型逐比特一致
            self.zero = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_control)])
            for z in self.zero:
                nn.init.zeros_(z.weight); nn.init.zeros_(z.bias)
            # 控制分支的**独立**输入嵌入。此前 c = x（分支与主干同起点），那样
            # 264.9 M 参数只是在扩容，不是第二条条件通路。现在它专吃 z_mosaic，
            # 把主干原生的 cond_image 通道让给「前一块」做块自回归（readme §3.6）。
            self.ctrl_embedder = _c.deepcopy(m.x_embedder)
            for p_ in self.ctrl_embedder.parameters(): p_.requires_grad = True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x_t, t, z_mosaic, y=None, gap=None, z_prev=None, z_mean=None,
                cfg_drop=None):
        """x_t / z_mosaic / z_prev: (B,16,T,H,W)   t,gap: (B,) in [0,1]

        两条条件通路，各走各的：
          z_mosaic -> ControlNet 分支（独立 embedder + zero-conv 注入）
          z_prev   -> 主干原生 cond_image 通道（块自回归，readme §3.6）
                      None 表示"没有前一块"，按 CTFlow 原做法置零 latent
                      （black_rate：既处理第一块的 OOD，也提供 CFG 的无条件分支）
        gap=None 或 0 -> 瞬时速度场 v_θ；gap>0 -> 区间 [t,t+gap] 的平均速度 u_θ

        cfg_drop: (B,) bool 或 None。True 的样本**跳过 ControlNet 的注入**，
          得到无条件分支 v_uncond。推理时 v = v_uncond + w·(v_cond − v_uncond)。

          ★ 为什么丢 ControlNet 而不是 z_prev：z_prev=0 已经有唯一含义
            ——「没有前一块」（第一块）。若 CFG 也用它当无条件分支，
            模型就分不清「我是第一块，请自由生成」和「这是 CFG 的基准前向」。
            而且纹理信息主要在 ControlNet 通路里，引导它才可能补纹理。
        """
        B = x_t.shape[0]
        if y is None:
            # ⚠️ v1~v9 这里喂的是**全零** —— 那对 caption 通路是 OOD。
            #    主干的 CaptionEmbedder 有一个学出来的 null embedding
            #    （buffer `y_embedding`, (1,768), norm 1.0003），那才是
            #    「没有 caption」的正确表示。全零主干从没见过。
            ye = getattr(getattr(self.net.model, "y_embedder", None), "y_embedding", None)
            y = (ye.to(x_t.dtype).view(1, 1, -1).expand(B, 1, -1) if ye is not None
                 else x_t.new_zeros(B, 1, self.net.config.caption_channels))
        if z_prev is None:
            z_prev = torch.zeros_like(x_t)                # CTFlow 的 zero_latent
        if self.mode == "cond":
            assert gap is None, "cond 模式暂不支持 gap"
            return self.net(x_t, t, encoder_hidden_states=y,
                            cond_image=z_mosaic, return_dict=False)[0]
        return self._forward_controlnet(x_t, t, z_mosaic, y, gap, z_prev, z_mean, cfg_drop)

    def _forward_controlnet(self, x_t, t, z_mosaic, y, gap=None, z_prev=None, z_mean=None,
                            cfg_drop=None):
        """复刻 STDiT.forward，在前 n_control 层注入可训控制分支。"""
        from einops import rearrange
        m = self.net.model
        x = torch.cat([x_t, z_prev], dim=1)               # 主干: 噪声态 + 前一块
        x = m.x_embedder(x)
        x = rearrange(x, "B (T S) C -> B T S C", T=m.num_temporal, S=m.num_spatial)
        x = x + m.pos_embed
        x = rearrange(x, "B T S C -> B (T S) C")

        if t.ndim == 0 or isinstance(t, (int, float)):
            t = torch.ones(x.shape[0], device=x.device) * t
        temb  = m.t_embedder(t, dtype=x.dtype)
        if gap is not None:
            temb = temb + self.gap_proj(self.gap_embedder(gap, dtype=x.dtype))
        t0    = m.t_block(temb)
        yy    = m.y_embedder(y.unsqueeze(1), self.training) if m.y_embedder is not None else None
        ylens = None
        if yy is not None:
            ylens = [yy.shape[2]] * yy.shape[0]
            yy = yy.squeeze(1).view(1, -1, x.shape[-1])

        # ── 控制分支的输入（32 通道）───────────────────────────────────
        # v4 及之前: cat([z_mosaic, z_mosaic]) —— 后 16 个通道是自我复制的**凑数**，
        #            白白浪费了一半输入带宽，而且分支得从头再推一遍 Deblocker
        #            已经算好的低频分解。
        # v5: cat([ẑ_mean, z_mosaic − ẑ_mean]) —— 预先分解成「已解释的」和
        #     「还没解释的」。两者相加即 z_mosaic，**信息一点不减**，参数量/
        #     显存/速度全部不变，纯赚。
        #
        # 为什么用 Deblocker 而不是小波做这个分解（实测 2026-09-01）：
        #   mosaic 的块网格周期 = 16 体素 ÷ f=8 = **2 latent 像素**，正好是
        #   latent 的 Nyquist 频率 —— 伪影和真实高频信号完全重叠，任何固定
        #   频带都分不开。而 Deblocker 是专门对这个伪影训出来的低通：
        #   周期-2 能量占比 z_mosaic 1.145% -> ẑ_mean 0.279%（z_GT 0.213%），
        #   去掉了 92.9% 的网格能量。
        if z_mean is None:
            z_mean = z_mosaic                       # 退化成 v4 口径（老 checkpoint 兼容）
            ctrl_in = torch.cat([z_mosaic, z_mosaic], dim=1)
        else:
            ctrl_in = torch.cat([z_mean, z_mosaic - z_mean], dim=1)
        c = self.ctrl_embedder(ctrl_in)
        c = rearrange(c, "B (T S) C -> B T S C", T=m.num_temporal, S=m.num_spatial)
        c = c + m.pos_embed
        c = rearrange(c, "B T S C -> B (T S) C")
        for i, blk in enumerate(m.blocks):
            tpe = m.pos_embed_temporal if i == 0 else None
            if i < getattr(self, "n_control", 0):
                c = self.ctrl[i](x=c, t=t0, y=yy, mask=ylens, tpe=tpe)
                inj = self.zero[i](c)
                if cfg_drop is not None:            # 无条件分支：跳过注入
                    inj = inj * (~cfg_drop).to(inj.dtype).view(-1, 1, 1)
                x = blk(x=x, t=t0, y=yy, mask=ylens, tpe=tpe) + inj
            else:
                x = blk(x=x, t=t0, y=yy, mask=ylens, tpe=tpe)
        return m.unpatchify(m.final_layer(x, temb))


def flow_batch(z_mean, z_gt, logit_normal=True, sigma=0.0):
    """构造 (x_t, t, target)。t ~ sigmoid(N(0,1))，readme §5.3。

    sigma > 0 时起点加噪：x0 = ẑ_mean + σ·ε   （SIFM 的 x₀ = m(x₁) + σζ；PMRF 同）

    ⚠️ 实测结论：σ 对本任务无益，默认保持 0。
    曾经的假设是「σ=0 时确定性 ODE 会收敛到条件均值，所以纹理有天花板」。
    2026-08-26 的步数扫描证伪了它——纹理随 T 单调增长到 32 都未饱和
    （L1: mean .387 → T1 .401 → T8 .451 → T32 .489），说明模型学到的是
    真实输运而非均值坍缩。σ=0.3 的对照跑反而让输出更靠近 ẑ_mean（补得更少）。
    保留这个参数只为可复现该对照，不建议开启。
    """
    B = z_mean.shape[0]
    x0 = z_mean if sigma == 0 else z_mean + sigma * torch.randn_like(z_mean)
    if logit_normal:
        t = torch.sigmoid(torch.randn(B, device=z_mean.device))
    else:
        t = torch.rand(B, device=z_mean.device)
    tb = t.view(B, *([1] * (z_mean.ndim - 1)))
    return (1 - tb) * x0 + tb * z_gt, t, z_gt - x0
