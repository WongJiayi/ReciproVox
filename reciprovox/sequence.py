"""Token layout shared by both directions.

The vocabulary is Qwen3's text vocabulary + structural tokens + 262,144 visual tokens
(one per atlas codebook entry). The order in which tokens are added MUST match training:
with_task=True gives v_offset = 151,938 and a total vocabulary of 414,144 (padded to x64).
"""
import numpy as np

CODEBOOK_SIZE = 262144
GRID = 16                       # 16 x 16 x 16 blocks of 16^3 voxels = 256^3 volume
V_OFFSET = 151938
BLACK = frozenset({0, 1, 2, 3, 17, 26, 435, 3979})   # background (air) codebook entries
BG_TOKEN = 26                                          # most frequent background block


def enc_u8(x, u8):  return [u8[int(x) & 0xFF]]
def enc_u16(x, u8): x = int(x) & 0xFFFF; return [u8[(x >> 8) & 0xFF], u8[x & 0xFF]]
def enc_u24(x, u8):
    x = max(0, min(int(x), 16777215))
    return [u8[(x >> 16) & 0xFF], u8[(x >> 8) & 0xFF], u8[x & 0xFF]]


def build_tokenizer(llm_path, with_task=True):
    """-> (tokenizer, special ids, field ids, u8 ids, task ids)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(llm_path, use_fast=True)
    sp = ["<start_of_volume>", "<end_of_volume>", "<start_of_slice>", "<end_of_slice>", "<next_row>"]
    fd = ["<H>", "<W>", "<D>", "<ROWLEN>", "<SLICELEN>", "<VOLLEN>"]
    u8 = [f"<u8_{i:03d}>" for i in range(256)]
    tok.add_tokens(sp + fd + u8)
    tk = ["<TASK_GEN>", "<TASK_REPORT>"] if with_task else []
    if tk: tok.add_tokens(tk)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    ids = lambda xs: {t: tok.convert_tokens_to_ids(t) for t in xs}
    return tok, ids(sp), ids(fd), [tok.convert_tokens_to_ids(t) for t in u8], ids(tk)


def volume_block(hb, wb, db, vt, v_off, sp, fd, u8):
    """Serialize a (db, hb, wb) grid of codebook ids into the LM token sequence."""
    ids = [sp["<start_of_volume>"]]
    ids += [fd["<H>"]] + enc_u8(hb, u8) + [fd["<W>"]] + enc_u8(wb, u8) + [fd["<D>"]] + enc_u8(db, u8)
    ids += [fd["<VOLLEN>"]] + enc_u24(hb * wb * db, u8)
    v = np.asarray(vt, dtype=np.int64); per = hb * wb
    for d in range(db):
        ids += [sp["<start_of_slice>"], fd["<SLICELEN>"]] + enc_u16(per, u8)
        for r in range(hb):
            ids += [fd["<ROWLEN>"]] + enc_u8(wb, u8)
            s = d * per + r * wb
            ids += (v[s:s + wb] + v_off).tolist()
            ids.append(sp["<next_row>"])
        ids.append(sp["<end_of_slice>"])
    ids.append(sp["<end_of_volume>"])
    return ids


VOLUME_HEADER_LEN = 1 + 2 + 2 + 2 + 4   # <start_of_volume> + H/W/D fields + VOLLEN
