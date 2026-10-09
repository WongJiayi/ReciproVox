"""Report -> CT.

  python scripts/generate_ct.py --text "No pleural effusion. ..." --out outputs/demo
  python scripts/generate_ct.py --reports reports.csv --out outputs/run --bs 8 --shard 0 --nshard 4

reports.csv needs a VolumeName column and Findings_EN (falls back to Impressions_EN).
Output per case:  <out>/tokens/<case>.npy   (16,16,16) codebook ids
                  <out>/<case>/frame_XXX.png  256 uint8 frames, training orientation
Resumable: finished cases are skipped. Token generation dominates the cost (~180 s/volume at bs=1;
bs=8 is ~8.5x faster); refinement is ~14 s/volume.
"""
import _bootstrap  # noqa
import os, csv, time, argparse, numpy as np, torch
from reciprovox.atlas import load_atlas, tokens_to_volume
from reciprovox.io import save_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="", help="CSV with VolumeName + Findings_EN")
    ap.add_argument("--text", default="", help="a single report instead of a CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--N", type=int, default=8, help="Rectifier steps")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--stage", type=int, default=2, choices=[0, 1, 2],
                    help="0 = raw atlas mosaic, 1 = + Deblocker, 2 = + Rectifier (default)")
    ap.add_argument("--lora", default=None, help="optional GRPO LoRA adapter dir")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    a = ap.parse_args()

    if a.text: cases = [("case_000", a.text)]
    else:
        cases = []
        for r in csv.DictReader(open(a.reports)):
            c = (r.get("VolumeName") or r.get("case_id") or "").replace(".nii.gz", "")
            t = (r.get("Findings_EN") or r.get("Impressions_EN") or r.get("report") or "")
            if c: cases.append((c, t))
    if a.n: cases = cases[:a.n]
    cases = cases[a.shard::a.nshard]
    os.makedirs(f"{a.out}/tokens", exist_ok=True)
    todo = [(c, t) for c, t in cases if not os.path.exists(f"{a.out}/{c}/.done")]
    print(f"[generate_ct] {len(todo)} / {len(cases)} cases to do", flush=True)

    need = [x for x in todo if not os.path.exists(f"{a.out}/tokens/{x[0]}.npy")]
    if need:
        from reciprovox.lm import ReciproVoxLM
        lm = ReciproVoxLM(lora=a.lora)
        g = torch.Generator(device="cuda"); g.manual_seed(a.seed)
        for k in range(0, len(need), a.bs):
            chunk = need[k:k + a.bs]; t0 = time.time()
            toks, st = lm.generate_volume_tokens([t for _, t in chunk], temp=a.temp, generator=g)
            for (c, _), t in zip(chunk, toks): np.save(f"{a.out}/tokens/{c}.npy", t)
            print(f"  tokens {k + len(chunk)}/{len(need)}  {time.time() - t0:.0f}s", flush=True)
        del lm; torch.cuda.empty_cache()

    atlas = load_atlas()
    ref = None
    if a.stage > 0:
        from reciprovox.refine import Refiner
        ref = Refiner(N=a.N, stride=a.stride)
    for c, _ in todo:
        vol = tokens_to_volume(np.load(f"{a.out}/tokens/{c}.npy"), atlas)
        if ref is not None: vol = ref(vol, seed=1234, stage=a.stage)
        save_frames(vol, f"{a.out}/{c}")
        open(f"{a.out}/{c}/.done", "w").close()
        print(f"  saved {c}", flush=True)


if __name__ == "__main__":
    main()
