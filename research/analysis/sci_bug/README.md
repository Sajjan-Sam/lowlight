# SCI bug reproduction

`sci_diagnose.py` runs SCI (own reimplementation in `enhancers.py`, official weights from
github.com/vis-opt-group/SCI `CVPR/weights/*.pt`) on the raw sample embedded in the progress document
(`image41.png`) and compares every variant with the doc's "SCI" sample (`image37.png`).

    python sci_diagnose.py <path/to/SCI/CVPR/weights> <dir with image41.png and image37.png>

Result (2026-11-03): the doc's SCI output matches **medium weights, eval mode, illumination map `i`**
(mean 64.4 vs 63.8, std 12.8 vs 12.8, slope 0.57 vs 0.57, mean |diff| 1.1/255).
The correct SCI output is `r = clamp(x / i, 0, 1)` (mean ≈ 143 here, always ≥ input).
Fix: save `r`, not `i`, then regenerate SCI images and re-run all SCI rows.
