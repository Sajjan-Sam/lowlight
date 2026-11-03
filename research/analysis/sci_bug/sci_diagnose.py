"""Find which SCI pipeline variant reproduces the thesis doc's SCI output (image37) from its raw input (image41)."""
import sys
import numpy as np
import torch
from PIL import Image
from enhancers import SCI

WEIGHTS, MEDIA = sys.argv[1], sys.argv[2]
raw = np.asarray(Image.open(f"{MEDIA}/image41.png").convert("RGB")).astype(np.float32) / 255
doc = np.asarray(Image.open(f"{MEDIA}/image37.png").convert("RGB")).astype(np.float32) / 255


def stats(a, tag):
    g, gr = a.mean(2), raw.mean(2)
    slope, off = np.polyfit(gr.ravel(), g.ravel(), 1)
    edge = np.abs(np.diff(g, axis=1)).mean()
    err = np.abs(a - doc).mean() * 255
    print(f"{tag:42s} mean={a.mean()*255:6.1f} std={g.std()*255:5.1f} slope={slope:5.2f} off={off*255:5.1f} "
          f"edge={edge*255:5.2f} min(out-in)={(a-raw).min()*255:7.1f} |diff vs doc|={err:5.1f}")


stats(raw, "RAW input (doc image41)")
stats(doc, "DOC 'SCI' output (image37)")
torch.set_grad_enabled(False)
for w in ["easy", "medium", "difficult"]:
    m = SCI(f"{WEIGHTS}/{w}.pt")
    for mode in ["eval", "train"]:
        m.eval() if mode == "eval" else m.train()
        for chan in ["RGB", "BGR"]:
            x = raw if chan == "RGB" else raw[..., ::-1].copy()
            i, r = m(torch.from_numpy(x).permute(2, 0, 1)[None])
            r = r[0].permute(1, 2, 0).numpy(); i = i[0].permute(1, 2, 0).numpy()
            if chan == "BGR":
                r, i = r[..., ::-1], i[..., ::-1]
            stats(r, f"{w}/{mode}/{chan} r (correct output)")
            if chan == "RGB":
                stats(i, f"{w}/{mode}/{chan} i (illumination map)")
