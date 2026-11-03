# Prior-art audit: "Does LLIE help object detection, and why?" (benchmark paper)

Audit date 2026-11-03. **Limits:** arxiv.org, ar5iv and most mirrors were blocked in this session (allow `arxiv.org` to fetch full texts), so I could not open any full text. All claims come from search-engine abstracts/snippets of the primary pages. The shared web-search budget (200 calls) ran out before I could check the feature-space and dehazing follow-ups. Anything marked **UNVERIFIED** needs a check against the PDF.

## A. Closest prior works

| Work | Venue / year / link | What they did | What they did NOT do |
|---|---|---|---|
| Wu, Lai, …, Shen, *Low-Light Enhancement Effect on Classification and Detection: An Empirical Study* | arXiv 2409.14461, Sep 2024. Preprint only, no venue found. arxiv.org/abs/2409.14461 | 12 LLIE methods (3 traditional, 9 deep). Classification on Caltech256-LL (synthetic) and Dark Class (real, 19 classes, 3,853 images). Detection on ExDark (one snippet: 3,000/1,800/2,563 split). Reports PSNR/SSIM/NIQE/BRISQUE. Retrains detectors on original vs enhanced data. Conclusion: the effect is "inconsistent, sometimes harmful", and performance is "comparable" with or without enhancement. | Which detectors and how many: **UNVERIFIED** (likely one YOLO). Size, lighting-condition, feature-space and seed analyses: none found. |
| Li, Zhao, Guo, *LIME-Eval* | arXiv 2410.08810, Oct 2024. ICLR 2025 submission, **withdrawn** (openreview.net/forum?id=hYU0P4Wlj9) | Argues that "fine-tune the detector on enhanced images" conflates the enhancer with the detector's adaptability. Retraining on images enhanced by Bread, IAT or Retinexformer (enhance before augmentation) did not beat raw dim ExDark (49.2 mAP); enhancing after augmentation helped. Proposes a label-free score: **frozen detectors pretrained on normal light** plus an energy score on their confidence maps. LIME-Bench: 6,362 human preference pairs from 750 users over 14 LLIE methods on ExDark images. Spearman ≈0.88 with mAP on synthetically darkened COCO-val (secondary source). | Only real-dark detection dataset is ExDark. No study across detector families, object sizes or lighting conditions, no mechanism analysis, no CIs. No follow-ups found. |
| Liu, Xu, Yang, Fan, Huang, *Benchmarking LLIE and Beyond* (VE-LOL) | IJCV 129:1153–1184, 2021 | Enhancement benchmark with full-reference, no-reference and "semantic" metrics. Face detection on VE-LOL-H (PyramidBox), plus a joint enhancement + face-detection model. | Generic multi-class detection; a multi-detector grid. |
| Yuan, Yang et al., *UG2+ Track 2* (DARK FACE) | arXiv 1904.04474. TIP 2020 (from the draft header, **venue partly verified**). Winning 2021 report: arXiv 2107.00818 | Haze, rain and low-light tracks. Cascades of enhancement then detection. Enhancement "not always" helps, and the best-SSIM method was not the best detector (DCPDN vs MSCNN). 2021 winner: MSRCR + Cascade R-CNN/DetectoRS. | Controlled grid, statistics, mechanism. |
| Loh & Chan, ExDark | CVIU 178:30–42, 2019 | 7,363 images, 12 classes, 10 lighting types. Feature-level analysis (hand-crafted and CNN): low-light effects "go deeper than illumination invariance". Proposes 2 metrics for judging enhancement for vision tasks. | Modern detectors; an enhancer × detector grid. |
| Li, Guo, …, Loy, LLIE survey (LLIV-Phone) | TPAMI 44(12):9396–9416, 2022 (arXiv 2104.10729) | **Frozen** DSFD (trained on WIDER FACE) on DARK FACE: every deep LLIE method *improved* face AP. Retinex-Net, Zero-DCE and TBEFN were the most robust. | Multi-class detection; detectors trained on dark data. Note: this is the opposite sign of our P2, because it uses a different protocol. |
| Sun, Ren, Wang, Cao, *Rethinking Image Restoration for Object Detection* | NeurIPS 2022 | Restoration (dehazing, LLIE) "cannot improve, sometimes reduces" detection. Fix: targeted adversarial pseudo-ground-truth, tested on one- and two-stage detectors. | Benchmark or explanation of *why*. |
| Same paradox, other degradations | Pei et al. ECCV 2018 (dehazing does not help classification). RESIDE/RTTS (arXiv 1712.04143, TIP 2019): task-driven criterion, PSNR/SSIM "insufficient". MPID deraining (CVPR 2019): deraining can hurt detection. URIE (ECCV 2020): recognition-friendly enhancer in front of a frozen model. VRD-IR (CVPR 2023): recognition-driven multi-degradation restoration. | These set the "restoration ≠ recognition" narrative. | Low light specifically, at scale. |
| *Two-stage object detection in low-light…* | PeerJ CS 11:e2799, 2025 | 3 LLIEs (ZeroDCE++, GLADNet, TBEFN) with YOLOv7 on ExDark. Enhancement *helped* (0.49→0.574). Claims NIQE "strongly correlates" with mAP (n=3 enhancers). | Any rigour. Easy to rebut, but you must cite it. |
| Kumari et al., systematic review + benchmark | Applied Soft Computing 201:115607, 2026 (Q1) | Review plus YOLOv8–v11 experiments on ExDark, DarkFace and LLVIP. | Whether it has an enhancer grid: **UNVERIFIED**. Only the YOLO family. |
| Li et al., *IQA: From Human to Machine Preference* (MPD) | CVPR 2025 | 2.25M machine-preference labels on 30k image pairs from 30 models. Human-centric IQA fails to predict machine preference. | Low light; detection mAP deltas. |
| Pham, *TaskGuard* | arXiv 2609.08011, Sep 2026 | Per-image predictor of whether a restoration will help a *frozen* detector. Reports −54.2% negative interventions on unseen degradations (Gaussian-trained, transfers to rain). A third-party review says its gain over a Detector+Residual baseline is not significant. | Low light (**UNVERIFIED**, probably absent). |
| Method papers that use enhance-then-detect baselines | FeatEnHancer (ICCV 2023), MAET (ICCV 2021), YOLA (NeurIPS 2024), FRBNet (NeurIPS 2025), FoCo (CVPR 2025), SGLDet (ICLR 2026), AMIEOD (arXiv 2605.06084), Retinexformer/MambaLLIE (YOLOv3 retrained on enhanced ExDark), LMOT (CVPR 2024: trains on low/well/all vs enhance-first) | They show enhancer-as-preprocessor tables, usually with one detector. | Analysis. |

## B. Gap check

**5(a) Train/test protocol matrices: PARTLY DONE.**
- 2409.14461 compares raw-trained with enhanced-trained detectors.
- LIME-Eval covers enhanced-trained detectors, the order of augmentation and enhancement, and frozen normal-light detectors.
- The TPAMI survey and SID-NISM test frozen pretrained detectors on enhanced images.
- LMOT trains on low, well-lit and mixed data.
- **Open:** a full matrix of {normal-light-pretrained, raw-trained, enhanced-trained, raw+enhanced mixed} × {raw, enhanced test} × several detector families × several datasets. No one has shown systematically that **the sign of ΔmAP flips with protocol** (frozen COCO detector: enhancement helps; raw-trained detector: it hurts).

**5(b) Object-size dependence: MOSTLY OPEN.**
- No dedicated LLIE study found.
- Scattered AP_S/M/L appears in method papers. LDWLE (Complex & Intelligent Systems 2024): enhancement "notably improves small objects" on COCO-d. NID-DETR (Sci. Rep. 2025): enhancement slightly hurts large objects.
- Our finding (small objects sometimes helped, medium/large hurt) agrees with these. It is novel as a systematic, multi-detector result, but it must cite them.

**5(c) Per-condition analysis on ExDark's 10 lighting types: OPEN.**
- Nothing found.
- MILL (arXiv 2511.15496, 2025/26) studies 11 controlled intensity levels, but for fidelity only, with no detection.

**5(d) IQA vs mAP correlation: PARTLY DONE.**
- LIME-Eval (its own metric vs mAP and vs human preference).
- PeerJ 2025 (NIQE, n=3).
- RESIDE and UG2+ (anecdotal disagreements).
- MPD (CVPR 2025, general machine-preference IQA).
- 2409.14461 reports IQA scores next to detection results; whether it computes a correlation is **UNVERIFIED**.
- **Open:** a multi-metric (PSNR/SSIM/LPIPS/NIQE/BRISQUE/MUSIQ/CLIP-IQA/LIME-Eval) × multi-detector × multi-dataset rank-correlation study with bootstrap CIs, plus partial correlations controlling for brightness.

**5(e) Feature-space / domain-shift explanation (CKA, MMD, FID in detector space): OPEN.**
- Papers only assert "distribution shift" in words.
- Loh & Chan analyse features of dark images, not of enhanced ones.
- Sun et al. 2022 use an adversarial framing.
- I found no CKA/MMD/linear-probe study (budget ran out before a second pass, so treat this as **likely open, not certain**).

**5(f) Multiple seeds / CIs: OPEN.**
- No LLIE-for-detection study reports them.
- This is expected hygiene rather than novelty, but it lets us say "differences below X mAP are noise". LIME+FRCNN +0.0043 almost certainly is noise.

**5(g) DETR-family vs CNN sensitivity: OPEN.**
- Only unsupported one-line claims in method papers.

**6. Task-driven / machine-oriented IQA that predicts ΔmAP: CONCEPT SCOOPED, LLIE-SPECIFIC VERSION OPEN.**
- Blind detection-quality predictors exist: Kong et al. (IEEE, 2019) and a LIVE study (2022).
- MPD (CVPR 2025) covers machine-preference IQA in general.
- LIME-Eval gives a label-free detector-energy score for ranking enhancers.
- TaskGuard (Sep 2026) predicts per-image restoration utility for frozen detectors.
- **Open:** a label-free predictor of per-image ΔmAP for LLIE, validated across detectors and datasets and benchmarked against LIME-Eval and TaskGuard.

**7. Largest enhancer × detector grid found:**
- 2409.14461: 12 enhancers × at least 1 detector, ExDark only.
- LIME-Eval: 14 enhancers (human study); detector count **UNVERIFIED**.
- TPAMI survey: about 10+ enhancers × 1 frozen face detector.
- Our **7 × 5 grid across 3 detector families** appears to be the widest *detector* coverage, but not the widest *enhancer* coverage.
- No grid across ExDark, DarkFace, LLVIP, BDD100K-night and NightOwls together was found.

## C. Verdict (blunt)

**Already scooped. Do not claim these as contributions:**
1. "LLIE does not help, or hurts, detection" (UG2+ 2019/20, Sun et al. NeurIPS 2022, 2409.14461, LIME-Eval).
2. "Retraining on enhanced images still does not beat raw" (LIME-Eval's 49.2 mAP result; 2409.14461). Our P3 replicates this.
3. "PSNR/SSIM/NIQE do not reflect machine utility" (RESIDE, UG2+, MPD).
4. "Evaluate LLIE with frozen pretrained detectors" (LIME-Eval, TPAMI survey).

Our current P2/P3 tables, on one dataset (ExDark with a custom split), with 7 enhancers, single runs, and a possible SCI pipeline bug, would read as a replication. That is not enough for Q1 and not enough for NeurIPS D&B.

**Still novel, if done properly:**
- Scale-dependent effect, across detectors and datasets.
- Per-lighting-condition effect on ExDark's 10 types.
- Detector-family sensitivity (one-stage CNN vs two-stage vs DETR), with seeds and CIs.
- Mechanistic explanation in feature space.
- Mixed-training protocol across detector families.

**Analyses that would make it clearly publishable:**

1. **Protocol-sign-flip matrix (the headline).** Same enhancers × ≥6 detectors (add YOLO11/12, RT-DETRv2, DINO/Co-DETR, and one open-vocabulary detector such as Grounding DINO) × 4 training regimes (COCO-pretrained frozen, raw-trained, enhanced-trained, mixed) × ≥4 datasets (ExDark *official* plus the custom split, DARK FACE, LLVIP-visible, BDD100K-night or NightOwls). Use 3 seeds, bootstrap CIs, and paired tests. Central claim: *"whether LLIE helps is a property of the train/test protocol, not of the enhancer."* No one has shown this systematically.

2. **Mechanism by controlled intervention plus feature analysis.** Split each enhancer's effect into components: (i) global brightness/gamma only (match mean intensity), (ii) colour/white-balance shift, (iii) noise amplification (enhancer followed by a denoiser, or measured local SNR), (iv) high-frequency/detail change. Then measure, per detector layer, CKA/MMD between raw and enhanced features. Regress ΔmAP on these factors per object, controlling for size, lighting condition and local SNR. This turns "it hurts" into "it hurts *because* X", and connects to the scale finding (noise amplification inside small boxes vs texture/contrast loss on large ones). Nobody has done this for LLIE.

3. **Label-free utility predictor plus oracle headroom.** (a) Compute an "oracle" per-image best enhancer, including "no enhancement", to quantify the headroom that selective enhancement (AMIEOD/TaskGuard-style) could ever reach. (b) Propose and validate a no-reference ΔmAP predictor, e.g. detector feature shift or confidence energy, against LIME-Eval, TaskGuard and 8+ IQA metrics, with rank correlations and CIs, across detectors and datasets. This links the benchmark to the DPI-Det method paper.

**Q1 journal (e.g. TIP, TCSVT, PR, ASOC, Neural Networks):** feasible with 1 + 2, multi-dataset coverage, seeds/CIs, and ≥10 enhancers including 2024–26 ones (diffusion, Mamba, FoCo, ISP-driven U-Net).

**NeurIPS D&B:** only feasible if we also ship a reusable artifact: a public toolkit and leaderboard (cached enhanced images, unified detector configs, metric scripts), *plus* new annotations. Examples: per-image illumination/SNR/lux metadata, verified lighting-condition labels, a cleaned standard ExDark split, or a small new paired dark detection test set. Without new data or a strong tool, D&B reviewers will call it "an analysis paper on existing datasets".

**Fix first:**
- Verify the SCI pipeline. Its output being darker than raw suggests a bug.
- Report results on a standard ExDark split for comparability.
- Treat sub-1-point differences as noise until seeds and CIs exist.

## Key links
arxiv.org/abs/2409.14461 · arxiv.org/abs/2410.08810 · openreview.net/forum?id=hYU0P4Wlj9 · flyywh.github.io/IJCV2021LowLight_VELOL · arxiv.org/abs/1904.04474 · arxiv.org/abs/1805.11227 · arxiv.org/abs/2104.10729 · proceedings.neurips.cc (Sun 2022, hash 1cac8326…) · openaccess.thecvf.com (VRD-IR CVPR23; MPD CVPR25; FoCo CVPR25; LMOT CVPR24; FeatEnHancer ICCV23) · ecva.net (URIE ECCV20; Pei ECCV18) · arxiv.org/abs/1712.04143 · peerj.com/articles/cs-2799 · sciencedirect.com/science/article/abs/pii/S1568494626010550 · arxiv.org/abs/2609.08011 · arxiv.org/abs/2609.13791 · arxiv.org/abs/2511.15496 · arxiv.org/abs/2605.06084 · iclr.cc/virtual/2026/poster/10009960
