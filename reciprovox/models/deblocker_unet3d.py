"""阶段一 D_φ — latent 空间的同形状 deblocking 回归器。

输入/输出均为 (B, 16, 16, 32, 32)，无上/下采样（readme §1.3：latent 网格进出一致）。
纯 L1 训练 -> 输出即后验均值 ẑ_mean。训完必须冻结（readme §10 原则 1）。

设计取舍:
  - 一个 atlas token 覆盖 2×2 个 latent cell（空间）× 16（z），所以最浅层的
    感受野必须 >= 2×2，用 3×3×3 卷积天然满足。
  - z 方向 16 帧全程不下采样到 1，保留块内 z 结构；空间 32->16->8。
  - 最低分辨率加自注意力，让远处解剖能互相参照（马赛克块内是常数，
    恢复内容必须靠上下文）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(c):
    return nn.GroupNorm(32 if c % 32 == 0 else 8, c)


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout, drop=0.0):
        super().__init__()
        self.n1 = norm(cin);  self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n2 = norm(cout); self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)   # 残差分支零初始化

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(self.drop(F.silu(self.n2(h))))
        return self.skip(x) + h


class AttnBlock3D(nn.Module):
    """在 (D,H,W) 展平后的全局自注意力。最低分辨率 4*8*8=256 个位置，代价可忽略。"""
    def __init__(self, c, heads=8):
        super().__init__()
        self.n = norm(c); self.heads = heads
        self.qkv = nn.Conv3d(c, c * 3, 1)
        self.proj = nn.Conv3d(c, c, 1)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, D, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.heads, C // self.heads, D * H * W).unbind(1)
        a = F.scaled_dot_product_attention(q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1))
        return x + self.proj(a.transpose(-2, -1).reshape(B, C, D, H, W))


class Down(nn.Module):
    """只在 H,W 上下采样，z 保持 16。"""
    def __init__(self, c):
        super().__init__(); self.op = nn.Conv3d(c, c, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
    def forward(self, x): return self.op(x)


class Up(nn.Module):
    def __init__(self, c):
        super().__init__(); self.op = nn.Conv3d(c, c, 3, padding=1)
    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest"))


class DeblockerUNet3D(nn.Module):
    def __init__(self, ch=16, base=112, mults=(1, 2, 3), nres=2, attn_at=(2,), drop=0.0):
        super().__init__()
        self.inp = nn.Conv3d(ch, base, 3, padding=1)
        chans = [base * m for m in mults]

        self.downs = nn.ModuleList(); skip_ch = [base]
        cur = base
        for i, c in enumerate(chans):
            for _ in range(nres):
                blk = [ResBlock3D(cur, c, drop)]
                if i in attn_at: blk.append(AttnBlock3D(c))
                self.downs.append(nn.Sequential(*blk)); cur = c; skip_ch.append(cur)
            if i < len(chans) - 1:
                self.downs.append(Down(cur)); skip_ch.append(cur)

        self.mid = nn.Sequential(ResBlock3D(cur, cur, drop), AttnBlock3D(cur), ResBlock3D(cur, cur, drop))

        self.ups = nn.ModuleList()
        for i, c in reversed(list(enumerate(chans))):
            for j in range(nres + 1):
                blk = [ResBlock3D(cur + skip_ch.pop(), c, drop)]
                if i in attn_at: blk.append(AttnBlock3D(c))
                if i > 0 and j == nres: blk.append(Up(c))
                self.ups.append(nn.Sequential(*blk)); cur = c

        self.out_n = norm(cur)
        self.out = nn.Conv3d(cur, ch, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # 初始为恒等：输出 = 输入

    def forward(self, x):
        """x: (B,16,16,32,32) -> 同形状。残差形式，初始即恒等映射。"""
        h = self.inp(x); hs = [h]
        for m in self.downs:
            h = m(h); hs.append(h)
        h = self.mid(h)
        for m in self.ups:
            h = m[0](torch.cat([h, hs.pop()], 1)) if isinstance(m[0], ResBlock3D) else m(h)
            for sub in m[1:]: h = sub(h)
        return x + self.out(F.silu(self.out_n(h)))


if __name__ == "__main__":
    m = DeblockerUNet3D()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 16, 16, 32, 32)
    with torch.no_grad(): y = m(x)
    print(f"参数量 {n/1e6:.1f} M")
    print(f"输入 {tuple(x.shape)} -> 输出 {tuple(y.shape)}")
    print(f"零初始化检查 max|y-x| = {(y-x).abs().max():.2e}  (应为 0)")
