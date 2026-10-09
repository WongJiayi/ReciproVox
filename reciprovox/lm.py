"""The bidirectional language model (Qwen3-8B backbone).

Two tasks, selected by a task token:
  <TASK_GEN>    report text  -> 4,096 visual tokens   (sampled, temperature 0.5)
  <TASK_REPORT> visual tokens -> report text          (greedy)
"""
import os, numpy as np, torch, torch.nn.functional as F
from .config import path
from .sequence import (CODEBOOK_SIZE, GRID, V_OFFSET, BLACK, VOLUME_HEADER_LEN,
                       build_tokenizer, volume_block, enc_u8, enc_u16)

QWEN3_PAD = 151643   # <|endoftext|>, used for left padding


class ReciproVoxLM:
    def __init__(self, ckpt=None, lora=None, base=None, device="cuda"):
        from transformers import AutoModelForCausalLM
        from safetensors.torch import load_file
        base = base or path("qwen3_base")
        self.tok, self.sp, self.fd, self.u8, self.tk = build_tokenizer(base, with_task=True)
        self.v_off = len(self.tok)
        assert self.v_off == V_OFFSET, f"v_offset={self.v_off}, expected {V_OFFSET}"
        nv = self.v_off + CODEBOOK_SIZE
        nv = ((nv + 63) // 64) * 64
        m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True, attn_implementation="sdpa")
        m.resize_token_embeddings(nv, mean_resizing=False)
        msg = m.load_state_dict(load_file(ckpt or path("lm_ckpt")), strict=False)
        miss = [k for k in msg.missing_keys if "rotary" not in k]
        if miss: raise RuntimeError(f"LM checkpoint is missing {len(miss)} keys: {miss[:5]}")
        m = m.to(device).eval()
        lora = lora if lora is not None else (os.environ.get("RECIPROVOX_LORA") or _cfg_lora())
        if lora:
            from peft import PeftModel
            m = PeftModel.from_pretrained(m, lora, adapter_name="a0", is_trainable=False)
            _verify_lora_B(m, lora, "a0")
            m = m.eval()
        self.m, self.device = m, device

    # ── report -> CT ─────────────────────────────────────────────────────
    def gen_prefix(self, report, max_text=512):
        return ([self.tk["<TASK_GEN>"]] + self.tok.encode(report, add_special_tokens=False)[:max_text]
                + volume_block(GRID, GRID, GRID, np.zeros(GRID ** 3, np.int64),
                               self.v_off, self.sp, self.fd, self.u8)[:VOLUME_HEADER_LEN])

    @torch.no_grad()
    def generate_volume_tokens(self, reports, temp=0.5, generator=None):
        """list[str] -> (B,16,16,16) int64 codebook ids, list[dict] stats. Different prompts are left-padded."""
        m, dev, sp, fd, u8 = self.m, self.device, self.sp, self.fd, self.u8
        prefix = [self.gen_prefix(r[:800]) for r in reports]
        B, Lm = len(prefix), max(len(p) for p in prefix)
        ids = torch.full((B, Lm), QWEN3_PAD, dtype=torch.long, device=dev)
        att = torch.zeros((B, Lm), dtype=torch.long, device=dev)
        for b, p in enumerate(prefix):
            ids[b, Lm - len(p):] = torch.tensor(p, device=dev); att[b, Lm - len(p):] = 1
        o = m(input_ids=ids, attention_mask=att, use_cache=True)
        st = dict(past=o.past_key_values, att=att)

        def feed(shared=None, per_seq=None):
            x = per_seq.view(B, 1) if per_seq is not None else \
                torch.tensor(shared, dtype=torch.long, device=dev).view(1, -1).expand(B, -1)
            st["att"] = torch.cat([st["att"], torch.ones((B, x.shape[1]), dtype=torch.long, device=dev)], 1)
            out = m(input_ids=x, past_key_values=st["past"], use_cache=True, attention_mask=st["att"])
            st["past"] = out.past_key_values
            return out.logits[:, -1, :].float()

        n = GRID
        toks = torch.zeros(B, n, n, n, dtype=torch.long, device=dev)
        ent = torch.zeros(B, n ** 3, device=dev); k = 0
        lo, hi = self.v_off, self.v_off + CODEBOOK_SIZE
        for d in range(n):
            lg = feed([sp["<start_of_slice>"], fd["<SLICELEN>"]] + enc_u16(n * n, u8))
            for r in range(n):
                lg = feed([fd["<ROWLEN>"]] + enc_u8(n, u8))
                for c in range(n):
                    v = lg[:, lo:hi]                         # restrict to the visual vocabulary
                    p0 = F.softmax(v, -1); ent[:, k] = -(p0 * (p0 + 1e-9).log()).sum(-1); k += 1
                    t = torch.multinomial(F.softmax(v / max(temp, 1e-6), -1), 1, generator=generator)[:, 0]
                    toks[:, d, r, c] = t
                    lg = feed(per_seq=t + lo)
                lg = feed([sp["<next_row>"]])
            lg = feed([sp["<end_of_slice>"]])
        out = []
        for b in range(B):
            f = toks[b].reshape(-1).cpu().numpy()
            out.append(dict(black=float(np.isin(f, list(BLACK)).mean()), uniq=int(np.unique(f).size),
                            ent=float(ent[b].mean())))
        return toks.cpu().numpy(), out

    # ── CT -> report ─────────────────────────────────────────────────────
    @torch.no_grad()
    def generate_reports(self, token_grids, max_new=384):
        """list of (4096,) codebook ids -> list[str]. Greedy decoding (do_sample=False)."""
        pre = [[self.tk["<TASK_REPORT>"]] + volume_block(GRID, GRID, GRID, np.asarray(t).reshape(-1),
                                                         self.v_off, self.sp, self.fd, self.u8)
               for t in token_grids]
        x = torch.tensor(pre, dtype=torch.long, device=self.device)   # fixed length, no padding needed
        out = self.m.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                              do_sample=False, pad_token_id=self.tok.eos_token_id,
                              eos_token_id=self.tok.eos_token_id)
        n = len(self.tok)
        return [self.tok.decode([t for t in row.tolist() if 0 <= t < n], skip_special_tokens=True)
                for row in out[:, x.shape[1]:]]


def _cfg_lora():
    from .config import CFG
    return CFG.get("lora", "")


def _verify_lora_B(m, adapter_dir, name):
    """peft can silently leave lora_B at zero (= no adapter at all). Check against disk and reload if so."""
    from peft.tuners.lora import LoraLayer
    from safetensors import safe_open
    layers = [(n, mod) for n, mod in m.named_modules() if isinstance(mod, LoraLayer)]
    zero = [(n, mod) for n, mod in layers if float(mod.lora_B[name].weight.float().norm()) == 0.0]
    if not zero: return
    with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), framework="pt") as h:
        keys = set(h.keys())
        for n, mod in zero:
            k = f"{n}.lora_B.weight"
            if k not in keys: raise RuntimeError(f"lora_B missing on disk for {n}")
            w = mod.lora_B[name].weight
            with torch.no_grad(): w.copy_(h.get_tensor(k).to(device=w.device, dtype=w.dtype))
    left = sum(1 for _, mod in layers if float(mod.lora_B[name].weight.float().norm()) == 0.0)
    assert left == 0, f"{left} lora_B layers still zero after reload"
