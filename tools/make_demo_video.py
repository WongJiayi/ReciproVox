"""Demo videos: synchronized tri-view (axial / coronal / sagittal) next to the report, input on the left.

  # report -> CT: the input report is shown in full while the generated volume scrolls
  python tools/make_demo_video.py --volume outputs/run/case_000 --report "..." --mode gen --out demos/x
  # CT -> report: the generated report is typed out while the input CT scrolls
  python tools/make_demo_video.py --volume frames/scan --report "<generated>" --reference "<ground truth>" \
      --mode report --out demos/y

Writes <out>.mp4 (1280x640, 24 fps) and <out>.gif (800x400, 10 fps). Volumes are PNG-frame dirs in
training orientation; they are displayed in radiological convention (head up, patient right on image left).
"""
import os, re, sys, argparse, subprocess, textwrap, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reciprovox.io import load_frames, u8_to_hu

W_, H_ = 1280, 640
BG, PANEL, FG, DIM = (13, 17, 23), (22, 27, 34), (230, 237, 243), (125, 133, 144)
ACCENT, HL = (88, 166, 255), (255, 166, 87)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
KEY = re.compile(r"(ground[- ]glass\w*|consolidation\w*|pleural effusion\w*|effusion\w*|emphysema\w*|nodul\w*|"
                 r"atelecta\w*|cardiomegaly|septal thickening\w*|bronchiectasis|fibrotic|sequela\w*|calcif\w*|"
                 r"atheroma\w*|ectatic|dilated|lymph nodes? (?:reaching|up to)[^.,;]*|mosaic attenuation|opacit\w*)", re.I)


NEG = re.compile(r"\b(no|not|without|neither|nor|negative)\b", re.I)


def window(h, lo=-1150, hi=250):
    return (np.clip((h - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def font(sz, bold=False): return ImageFont.truetype(FONT_B if bold else FONT, sz)


def draw_text(d, text, x, y, w, fnt, color=FG, hl=True, lh=None, max_y=None):
    """Word-wrapped text with highlighted findings. Returns the next y."""
    lh = lh or int(fnt.size * 1.42)
    spans = []
    if hl:                                   # highlight only findings in clauses without a negation
        for c in re.finditer(r"[^.;]+[.;]?", text):
            if NEG.search(c.group()): continue
            spans += [(c.start() + m.start(), c.start() + m.end()) for m in KEY.finditer(c.group())]
    words = [(m.start(), m.group()) for m in re.finditer(r"\S+", text)]
    cx, cy = x, y
    for s, wd in words:
        ww = d.textlength(wd + " ", font=fnt)
        if cx + ww > x + w and cx > x: cx, cy = x, cy + lh
        if max_y and cy > max_y: d.text((cx, cy), "…", font=fnt, fill=DIM); return cy + lh
        on = any(a <= s < b for a, b in spans)
        d.text((cx, cy), wd, font=fnt, fill=HL if on else color)
        cx += ww
    return cy + lh


def frame(vol, z, report, mode, reveal=1.0, reference="", title=""):
    T, H, W = vol.shape
    im = Image.new("RGB", (W_, H_), BG); d = ImageDraw.Draw(im)
    # header
    d.text((28, 18), "ReciproVox", font=font(24, True), fill=FG)
    tag = "Report  →  CT" if mode == "gen" else "CT  →  Report"
    d.text((28 + d.textlength("ReciproVox", font=font(24, True)) + 18, 23), tag, font=font(17), fill=ACCENT)
    if title: d.text((W_ - 28 - d.textlength(title, font=font(14)), 25), title, font=font(14), fill=DIM)
    # images: axial 480 + coronal / sagittal 236. Input on the left, output on the right:
    # report -> CT puts the report first, CT -> report puts the CT first.
    y0, A, S = 72, 480, 236
    PW = W_ - 56 - (A + 8 + S) - 20                 # report panel width
    x0 = 28 + PW + 20 if mode == "gen" else 28
    ax = Image.fromarray(window(vol[z])).resize((A, A), Image.BILINEAR).convert("RGB")
    co = Image.fromarray(window(vol[:, H // 2, :])).resize((S, S), Image.BILINEAR).convert("RGB")
    sa = Image.fromarray(window(vol[:, :, W // 2])).resize((S, S), Image.BILINEAR).convert("RGB")
    im.paste(ax, (x0, y0)); im.paste(co, (x0 + A + 8, y0)); im.paste(sa, (x0 + A + 8, y0 + S + 8))
    zy = int((z + 0.5) / T * S)
    for yy in (y0 + zy, y0 + S + 8 + zy):
        d.line([(x0 + A + 8, yy), (x0 + A + 8 + S, yy)], fill=ACCENT, width=2)
    lab = font(12)
    def tag_(x, y, t):
        d.rounded_rectangle([x - 5, y - 3, x + d.textlength(t, font=lab) + 5, y + 16], radius=4, fill=(0, 0, 0))
        d.text((x, y), t, font=lab, fill=FG)
    tag_(x0 + 10, y0 + 8, f"axial  {T - z:3d}/{T}")
    tag_(x0 + A + 18, y0 + 8, "coronal")
    tag_(x0 + A + 18, y0 + S + 16, "sagittal")
    d.text((x0, y0 + A + 12), "R", font=lab, fill=DIM); d.text((x0 + A - 10, y0 + A + 12), "L", font=lab, fill=DIM)
    # report panel
    px = 28 if mode == "gen" else x0 + A + 8 + S + 20; pw = PW
    d.rounded_rectangle([px, y0, px + pw, y0 + A], radius=10, fill=PANEL)
    ix, iw = px + 18, pw - 36
    if mode == "gen":
        d.text((ix, y0 + 14), "INPUT REPORT", font=font(12, True), fill=ACCENT)
        draw_text(d, report, ix, y0 + 40, iw, font(14), max_y=y0 + A - 30)
    else:
        n = int(len(report) * reveal)
        d.text((ix, y0 + 14), "GENERATED REPORT", font=font(12, True), fill=ACCENT)
        yb = draw_text(d, report[:n] + ("▌" if reveal < 1 else ""), ix, y0 + 40, iw, font(14),
                       max_y=y0 + (300 if reference else A - 30))
        if reference:
            yr = max(yb + 10, y0 + 316)
            d.line([(ix, yr), (ix + iw, yr)], fill=(48, 54, 61), width=1)
            d.text((ix, yr + 10), "REFERENCE (radiologist)", font=font(11, True), fill=DIM)
            draw_text(d, reference, ix, yr + 32, iw, font(11), color=DIM, hl=False, max_y=y0 + A - 24)
    d.text((28, H_ - 34), "highlighted = findings", font=font(11), fill=HL)
    return im


def encode(frames, out, fps):
    w, h = frames[0].size
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                          "-crf", "18", "-movflags", "+faststart", out + ".mp4"], stdin=subprocess.PIPE)
    for f in frames: p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    pal = "fps=10,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".mp4", "-vf", pal, "-loop", "0", out + ".gif"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", required=True); ap.add_argument("--report", required=True)
    ap.add_argument("--reference", default=""); ap.add_argument("--mode", choices=["gen", "report"], default="gen")
    ap.add_argument("--title", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=24); ap.add_argument("--step", type=int, default=2)
    a = ap.parse_args()
    vol = u8_to_hu(load_frames(a.volume))[::-1, :, ::-1]            # head up, radiological left/right
    T = vol.shape[0]
    zs = list(range(int(T * 0.06), int(T * 0.94), a.step)); zs = zs + zs[::-1]
    frames = [frame(vol, z, a.report, a.mode, reveal=min(1.0, 1.6 * i / len(zs)), reference=a.reference,
                    title=a.title) for i, z in enumerate(zs)]
    frames += [frames[-1]] * a.fps                                  # hold the last frame for 1 s
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    encode(frames, a.out, a.fps)
    frames[len(zs) // 4].save(a.out + ".png")
    print("->", a.out + ".mp4 / .gif / .png", len(frames), "frames")


if __name__ == "__main__":
    main()
