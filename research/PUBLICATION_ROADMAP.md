# Publication Roadmap: Low-Light Enhancement and Object Detection (MS thesis)

*Deep-dive review of `MS_22255_RJS (2).docx`, prepared 2026-11-03.*

**Inputs:**
- The full progress document: text, every table, and all 41 figures.
- A reanalysis of the reported numbers: `analysis/`.
- Four parallel research audits: `literature/01–04`:
  - prior benchmark studies
  - prior methods, including a component-by-component novelty check of DPI-Det
  - alternative directions: VLMs, captioning, distillation, test-time adaptation
  - a reviewer-style methodology audit, plus venues and deadlines

> **Verification caveat.** This session's network policy blocks arxiv.org, OpenReview, CVF, NeurIPS proceedings and Semantic Scholar. The agents therefore checked papers through search-engine snippets and authors' GitHub pages, not full PDFs. Items marked UNVERIFIED in `literature/` must be checked against the papers before you cite them or claim novelty.

---

## 0. TL;DR

**Realistic output: 3 distinct papers, plus a 4th if time allows.**

| # | Paper | Type | Target | Probability if executed well |
|---|---|---|---|---|
| **A** | *Brighter Is Not Better*: a protocol-controlled, mechanistic study of when and why low-light enhancement helps or hurts detection | Benchmark + analysis | **Q1 journal** (Pattern Recognition / TCSVT; IJCV as a reach) | High |
| **B** | *Utility-Gated Cross-Illumination Distillation*: a detector that runs on raw images and learns from enhanced views only on the objects where enhancement actually helps | Method | **A\* conference** (ACM MM 2027 or AAAI-28; ICCV 2027 only if results are ready by February) | Medium |
| **C** | *Enhance for What, Not Where*: open-vocabulary detectors and VLMs in the dark, plus a cross-view class-presence prior | Benchmark + method (multimodal) | **A\*** (ACM MM 2027 / NeurIPS 2027 Evaluations & Datasets track) or a Q1 journal | Conditional on a 2-week pilot |
| D | *No-Regret Detection-Preserving Enhancement* (DPI-Det redesigned) | Method with image output | **Q1 journal** (TCSVT / PR / TMM) | Medium; do this after B |

Rules to keep:
- **Do not split further.** More slices of the same results invite desk rejects for incremental, "salami" publication.
- **DPI-Det as currently written is not publishable at an A\* venue.** Section 5 explains why; Paper D is the version that is.
- **Two of today's headline results may be wrong** (Section 3). Fix them before writing anything.

---

## 1. What the document contains: what is done and what is not

| Item | Status |
|---|---|
| ExDark set-up: 7,363 images, 12 classes, custom 70/20/10 split (5,154 / 1,472 / 737), YOLO and COCO formats | ✅ Done |
| 5 detectors trained on raw ExDark: YOLOv8n, YOLOv10n, RT-DETR, Faster R-CNN, Deformable DETR | ✅ Done (1 seed each) |
| 7 enhancers run on all images: LIME, Zero-DCE++, SCI, Retinexformer, SNR-Aware, PairLIE, KinD++ | ✅ Done (SCI suspect, see §3) |
| P1 raw→raw and P2 raw-trained→enhanced test, 7 enhancers × 5 detectors (mAP50, mAP50-95, P, R, F1) | ✅ Done |
| P3 enhanced→enhanced | ⚠️ Partial: YOLOv8n only, 5 of 7 enhancers |
| Per-class ΔAP and object-size ΔAP | ⚠️ YOLOv8n only |
| Qualitative grids (raw, 7 enhancers, ground truth) for 3 detectors | ✅ Done |
| Literature-survey slides (RetinexNet, LIME) | ⚠️ Started; the survey table is mostly empty |
| P4 mixed training, P5 proposed method | ❌ Not done |
| Diff-Retinex enhancer | ❌ Not done |
| LLVIP-visible, DarkFace, BDD100K-night | ❌ Not done (planned) |
| Image-quality metrics (PSNR, SSIM, LPIPS, NIQE, BRISQUE, LOE, entropy, exposure) and their correlation with mAP | ❌ Not done (table empty) |
| ExDark per-lighting-condition AP (10 conditions) | ❌ Not done |
| Seeds, confidence intervals, significance tests | ❌ Not done |
| FPS, FLOPs, parameter counts | ❌ Not done |
| DPI-Det design: equations, 8 losses, 4 modules, training stages | 📝 Design only, no code or results |
| Whiteboard idea: several enhancers → captioner per view → detectors → combine ("text-guided OD") | 📝 Sketch only |
| Thesis Objective 2 | ❌ Blank. A proposed wording is in §8. |

---

## 2. What the existing numbers already show

This is a reanalysis of the reported tables. Script and output are in `analysis/`.

1. **Harm is a property of the enhancer, not the detector.** The enhancer ranking is almost identical across all 5 detectors: Kendall's W = 0.974 (p = 1e-4), mean pairwise τ = 0.92. So harm can probably be predicted from the image alone. This motivates a label-free utility predictor (Papers A and B).
2. **Localization suffers more than recognition.** In relative terms mAP50-95 drops 1.2–1.5× more than mAP50 for every enhancer. Enhancement damages box precision (edges, boundaries) more than class evidence. This hints at the "enhance for *what*, not *where*" hypothesis behind Paper C.
3. **Most of the penalty is recoverable domain shift; part is real information loss.** For YOLOv8n, split the total drop (P1 − P2) into a shift part (P3 − P2) and an irreversible part (P1 − P3):

   | Enhancer | Total drop | Domain shift (recoverable) | Irreversible loss |
   |---|---|---|---|
   | LIME | 0.0099 | 83% | 17% |
   | Zero-DCE++ | 0.0407 | 90% | 10% |
   | Retinexformer | 0.0657 | 81% | 19% |
   | SNR-Aware | 0.0625 | 58% | **42%** |
   | SCI | 0.1440 | 62% | 38% (suspect) |

   This fits the data-processing inequality: a deterministic enhancer f cannot add label information, so I(Y; f(X)) ≤ I(Y; X). Enhancement can only help through inductive bias: moving the input closer to the detector's pretraining distribution, or helping when training data is scarce. That is a theory hook for Paper A.
4. **The effect depends on object size.**
   - Small objects sometimes gain: Zero-DCE++ +18% and KinD++ +15% relative to small-object AP.
   - Medium and large objects lose with every enhancer, by 6–32% relative.
   - Caveat: small-object AP is only 0.126 raw, so the small-object gains need confidence intervals before they count.
5. **Detector families differ in sensitivity.** Mean relative drop across enhancers:

   | Detector | Mean relative drop |
   |---|---|
   | Deformable DETR | −13.6% (most fragile) |
   | YOLOv8n | −10.7% |
   | YOLOv10n | −9.1% |
   | RT-DETR | −8.8% |
   | Faster R-CNN | −7.5% (most robust; −4.3% excluding SCI) |

6. **Both precision and recall fall.** For SCI, PairLIE, Retinexformer and KinD++ the recall loss is the larger one: enhanced images make objects disappear more than they create false positives. A TIDE-style error breakdown would make this concrete.
7. **Several reported differences are inside the noise.** On a 737-image test set the 95% interval on a paired mAP50 difference is roughly ±0.01–0.015 (estimated). The doc itself shows this:
   - YOLOv8n raw appears as both 0.7318 and 0.7332.
   - LIME on RT-DETR is +0.031 on the 74-image subset but −0.003 on the full set.

   So LIME's −0.0055 average, the single positive cell (LIME + Faster R-CNN, +0.0043), and "P3 still below raw" for LIME and Zero-DCE++ are **not yet claimable**.

---

## 3. Fix these before writing anything

| Severity | Problem | Fix |
|---|---|---|
| 🔴 Fatal | **The SCI output is almost certainly wrong.** In the official SCI code the illumination is `i = clamp(x + σ(f(x)), 1e-4, 1) ≥ x` and the saved output is `r = clamp(x/i)`. A correct SCI output is therefore never darker than the input, with local gain ≥ 1. The doc's SCI samples are washed out and darker than raw in places: per-pixel slope 0.24–0.80 with an offset of about 40, edges about 40% weaker. Every other enhancer has slope ≥ 1. The likely cause is that `i` (or a wrongly normalised tensor) was saved instead of `r`, or that wrong weights or wrong BGR/[0,255] handling were used. | Re-run SCI from the official repo. Check `r ≥ x` holds pixel-wise on 20 images. Re-run P2 and P3 for SCI. Until then, treat "SCI is worst (−0.149)" as invalid. |
| 🔴 Fatal | **No variance.** One seed per detector and no confidence intervals. | Use 3 seeds for at least YOLOv8n, RT-DETR and Faster R-CNN. Save **per-image predictions** for every run. Report paired bootstrap 95% CIs over test images, Holm-corrected paired tests, and Friedman / critical-difference diagrams for the enhancer ranking. |
| 🟠 Major | **P2 is unfair by construction.** Reviewers will say a raw-trained model is obviously worse on a shifted input. | Add **P0**: COCO-pretrained detectors run zero-shot on raw vs enhanced images. All 12 ExDark classes map 1-to-1 to COCO (People→person, Motorbike→motorcycle, Table→dining table, …), so this costs **no training**. P0 is where enhancement *should* help. The TPAMI 2022 survey found that enhancement helped a frozen face detector, so a sign flip between P0 and P1/P2 is likely and would be a headline result. |
| 🟠 Major | **Custom split, not comparable to prior work.** | Also report the **MAET 80/20 split** used by IA-YOLO, MAET, IAT, EMV-YOLO and AMIEOD (YOLOv3 baseline ≈ 76.4 mAP50), and/or the YOLA/FRBNet protocol. Keep the 70/20/10 split as a secondary result. |
| 🟠 Major | **Pipeline confounds**: resizing, JPEG re-encoding, colour order. | Add an **identity control**: raw images passed through the exact same load/save path as the enhancers. Add a **brightness-matched control**: a global gamma that matches each enhancer's mean luminance. Together they separate "brightening" from "everything else the enhancer does". |
| 🟡 Minor | P3 covers only YOLOv8n and 5 enhancers; per-class and size results only YOLOv8n. | Extend to at least 3 detectors and all enhancers. |
| 🟡 Minor | **Factual errors in the doc:** <ul><li>The LLVIP citation links to LIME's DOI.</li><li>The ExDark "640×360, mean 9.84" figures describe one image, not the dataset.</li><li>The LoL-v2 row says real+synthetic, but its counts are Real only.</li><li>KAIST/FLIR and NightOwls/Dark Zurich are merged into single rows.</li><li>The BDD100K description is wrong.</li><li>Faster R-CNN + LIME has an mAP50-95 identical to raw (check for a copy-paste error).</li><li>Every reference URL carries `?utm_source=chatgpt.com`.</li></ul> | Re-verify every citation by hand from the primary source before it goes into a paper. |

---

## 4. Already published (do not claim) vs still open

**Already published. Cite these, do not claim them as new:**
- "LLIE does not help, or hurts, detection": UG2+ (2019/20); Sun et al., *Rethinking Image Restoration for Object Detection*, NeurIPS 2022; arXiv 2409.14461 (12 enhancers on ExDark); LR-YOLO, NeurIPS 2025; a CVPRW 2026 LoViF paper.
- "Retraining on enhanced images still loses to raw": **LIME-Eval** (arXiv 2410.08810) shows this on ExDark with Bread, IAT and Retinexformer. Your P3 alone is a replication.
- "PSNR/NIQE do not reflect machine utility": RESIDE, UG2+, and *IQA: From Human to Machine Preference* (CVPR 2025).
- "Evaluate enhancers with frozen detectors": LIME-Eval and the TPAMI 2022 survey.
- "Predict per image whether restoration will help a detector": **TaskGuard** (arXiv 2609.08011, Sep 2026), mostly non-low-light (UNVERIFIED); PreActIR (2610.20384). This area is filling fast.

**Still open.** No exact match was found, checked via snippets only:

| Gap | Used in |
|---|---|
| Full protocol matrix {frozen COCO, raw-trained, enhanced-trained, mixed} × detector families × datasets, showing the sign of the effect flips with protocol | A |
| Split of the penalty into domain shift vs irreversible information loss, by size, lighting condition and detector family | A |
| Enhancement benefit as a function of training-data size and backbone freezing. *Not found in one targeted search; verify.* | A |
| Mechanism by controlled intervention (gain-only, colour-only, noise-only, detail-only) plus layer-wise CKA/MMD feature shift, regressed against per-object ΔAP | A |
| Systematic object-size effect across detectors and datasets. Only scattered hints exist (LDWLE 2024; NID-DETR 2025). | A, B |
| Analysis across ExDark's 10 lighting conditions | A |
| Per-object, utility-gated distillation from enhanced views into a raw-input detector. SGLDet (ICLR 2026), RMD-Net, Clean Feature Distillation, KDFA and TorchAdapt all transfer *without* a per-object gate. | B |
| Zero-shot open-vocabulary detectors and VLMs on real low-light boxes × enhancers × object size; captions or class presence across enhanced views as a detection prior | C |

---

## 5. Verdict on DPI-Det as written

**Rejected at CVPR, ICCV or ECCV with high likelihood.** Every component exists already. Novelty scores out of 10 are in `literature/02`:

| Component | Already done by | Novelty |
|---|---|---|
| Retinex-SNR encoder | DAI-Net, Trash-to-Treasure, EMV-YOLO, YOLA | 2 |
| W = 1 + A gain map | Zero-DCE, IA-YOLO/GDIP, SimROD | 2 |
| Uncertainty gate | GDIP, ICPR 2020, AMIEOD; U has no supervision as written | 3 |
| FiLM adapter | RAW-Adapter, Trash-to-Treasure | 3 |
| Masked feature-consistency loss | DAI-Net, SMM, FoCo | 2 |
| DCT loss | FRBNet, SGLDet, DEFormer | 1–2 |
| Freeze-then-finetune training | IA-YOLO, AMIEOD | 2 |
| **Whole combination** | — | **3** |

Specific technical flaws reviewers will hit:
1. **It may learn to do nothing.** Three pressures pull W toward 1: L_feat matches features to the *raw* image under a raw-trained detector, L_id keeps the background identical to raw, and raw is already the best input according to your own P2 and P3 results.
2. **Train–test mismatch and label leakage.** M_obj comes from ground-truth boxes in training but from proposals at test time. Box-shaped brightening also leaks the labels.
3. **The physics doesn't apply.** S = μ/√(αμ+β) is a raw-sensor Poisson-Gaussian model. ExDark images are post-ISP, gamma-encoded JPEGs, so with α and β learnable it is no longer physics. Even SNR-Aware's official code uses a blur heuristic, not this model.
4. **The losses have no target.** R/L decomposition and L_ret have no ground truth because ExDark has no paired normal-light images.
5. **Too many parts.** Eight losses and four modules make the ablations explode, and reviewers will put the gains down to extra capacity or extra fine-tuning epochs.
6. **Heavy overlap with prior work.** Adaptive detection-driven enhancement is already taken by ICPR 2020, IA-YOLO/GDIP and **AMIEOD** (2026, ExDark 82.1 vs a 76.5 baseline).

What to keep: the *intent*, enhancement constrained by what the detector needs. That becomes Papers B and D.

---

## 6. The papers in detail

### Paper A: *Brighter Is Not Better: When and Why Low-Light Enhancement Helps or Hurts Object Detection*
**Type:** benchmark + mechanism. **Target:** Q1 journal: Pattern Recognition, IEEE TCSVT, IJCV (reach; it published the VE-LOL benchmark), Image & Vision Computing, EAAI. Not JVCIR, which is not Q1.

**Core claim:** whether enhancement helps detection depends on the **training protocol and object scale**, not on the enhancer's image quality. Most of the penalty is recoverable distribution shift. The rest is irreversible information loss that label-free image and feature statistics can predict.

**Contributions and experiments:**
1. **Protocol matrix P0–P4:**
   - P0: frozen COCO-pretrained, zero-shot
   - P1: raw → raw
   - P2: raw-trained, tested on enhanced
   - P3: enhanced → enhanced
   - P4: mixed raw + enhanced training

   Scope: ≥5 detectors (YOLOv8n/s, YOLO11, RT-DETR, Faster R-CNN, Deformable DETR or DINO), 8–10 enhancers (add 1–2 recent ones, e.g. a 2024–26 diffusion or Mamba enhancer), and ≥3 datasets: ExDark (MAET split + yours), DarkFace, and one of LLVIP-visible / BDD100K-night / NOD. 3 seeds. **Headline:** the sign flips between P0 and P1/P2.
2. **Penalty decomposition** into shift vs irreversible loss, per enhancer, size, lighting condition and detector family, with the data-processing-inequality framing.
3. **Scaling test:** train on 10 / 25 / 50 / 100% of the training set, with full fine-tuning vs a frozen backbone. Prediction: enhancement helps only in the low-data or frozen regime, and the benefit shrinks as the detector's capacity to adapt grows.
4. **Mechanism by intervention.** Rebuild each enhancer's output from its parts:
   - global gain/gamma only, matched to the enhancer's mean luminance
   - colour transfer only
   - noise residual only
   - high-frequency detail only

   Measure ΔAP for each part. Compute layer-wise CKA/MMD between raw and enhanced backbone features. Regress per-object ΔAP on size, local luminance, local noise σ, ΔE colour shift, contrast change and CKA. This should explain *why* small objects gain while large ones lose.
5. **Conditional effects:** size (S/M/L), **ExDark's 10 lighting conditions**, class, and detector family. Add a TIDE error breakdown.
6. **Image quality ≠ utility:** Spearman/Pearson correlation of NIQE, BRISQUE, MUSIQ, CLIP-IQA, LOE, entropy and exposure error with ΔmAP across all (enhancer × detector × dataset) cells, with bootstrap CIs. Compare against the LIME-Eval score.
7. **Oracle headroom:** the mAP of picking the best of {raw, each enhancer} per image. This gives an upper bound for any selection method (AMIEOD, TaskGuard) and links to Paper B.
8. **Release:** cached enhanced images, per-image predictions, unified configs, scripts.

**Compute:** about 600 GPU-hours (a lean version is about 300) on one RTX-class GPU.

**Must cite and position against:** 2409.14461, LIME-Eval, the TPAMI 2022 survey, Sun et al. (NeurIPS 2022), VE-LOL (IJCV 2021), UG2+, CVPR 2025 human-vs-machine IQA, TaskGuard, LR-YOLO.

**Upgrade to NeurIPS 2027 Evaluations & Datasets (A\*)** only if you also release a reusable artifact: a toolkit and leaderboard, plus new annotations such as verified lighting-condition and per-image noise labels for ExDark test, or a cleaned standard split.

---

### Paper B: *Learn From the Light Only Where It Helps: Utility-Gated Cross-Illumination Distillation for Low-Light Detection*
**Type:** method. **Target:** A\*: ACM MM 2027 (~Apr 1, 2027, estimate) or AAAI-28 (~late Jul 2027, estimate); ICCV 2027 (~early Mar 2027, estimate) only if strong by mid-February. Fallback: TCSVT / TIP / PR.

**Insight, taken from your own data:** enhancement's usefulness differs from object to object. It helps some (small, very dark), hurts most (medium and large), and most of the harm is a pixel-level distribution shift. So don't *enhance the input*. Instead, **move the useful information from enhanced views into a raw-input detector, selectively, during training only.**

**Method sketch:**
- **Teachers:** the same detector fine-tuned on each enhanced domain (you already have the P3 YOLOv8n models), plus the raw model.
- **Per-object utility:** for each ground-truth object *o* and teacher *k*, compute u_k(o) = L_det^raw-teacher(o) − L_det^k-teacher(o), i.e. how much better teacher *k* does on *o* than the raw teacher. The gate g_k(o) is a softmax or margin-thresholded argmax over {raw, k}, with **raw as the default**.
- **Student:** sees raw images only. Its loss is L_det plus the gated distillation term Σ_o Σ_k g_k(o)·L_KD(o), where L_KD is distillation of features and logits inside *o*'s region on the feature-pyramid level assigned to *o*'s scale. Background is distilled from the raw teacher.
- **Optional:** a small label-free utility predictor, which allows semi-supervised training on unlabeled dark images (DarkFace has about 9k unlabeled).
- **Zero inference overhead:** the deployed detector is a plain raw-input YOLOv8n or RT-DETR.

**Closest prior work and the difference:**
- SGLDet (ICLR 2026), RMD-Net, Clean Feature Distillation, KDFA and TorchAdapt (ICCV 2025) transfer from enhanced or normal-light views **uniformly**.
- FGD-style box-masked distillation is not illumination-aware.
- AMIEOD selects **per image** at inference.
- Paper B's difference: **per-object, per-teacher, ground-truth-verified utility gating, with raw as the default, from several heterogeneous enhancer teachers, at no inference cost.**

**Baselines:**
- Raw.
- Raw + photometric augmentation (gamma, brightness, noise).
- Raw fine-tuned for the same extra epochs.
- P3 and P4.
- Ungated distillation (all objects) and a random gate.
- SGLDet, YOLA, FeatEnHancer, FRBNet, PE-YOLO/IA-YOLO, DAI-Net, and AMIEOD if its code is available.

**Ablations:** gate type, number and choice of teachers, which pyramid levels, size-wise AP.

**Bar for acceptance:** about +1.5–2 mAP50 over the strongest baseline on ≥2 datasets, with 3 seeds. **Compute:** about 700 GPU-hours.

**Main risk:** gains on ExDark under 1 mAP. Mitigate by showing consistent gains across 3–4 datasets and on small or very dark objects.

---

### Paper C: *Enhance for What, Not Where: Open-Vocabulary Detection and VLMs in the Dark*
This is the strongest version of the whiteboard captioning idea. **Run a 1–2 week pilot first.**

**Hypothesis:** enhanced views improve *recognition* (which classes are present) even while they hurt *localization*. Finding 2 in §2 is a hint.

**Pilot, inference only:** on the 737 test images × 8 views (raw + 7 enhancers), measure:
- image-level class-presence F1 from a VLM (Qwen2.5-VL / Florence-2 / InternVL) and from the detectors;
- the hallucination rate per view.

**If enhanced views recognise presence better while box AP falls, the story works:**
- **Benchmark part:** zero-shot Grounding DINO, YOLO-World, OWLv2 and Florence-2 × enhancers × datasets × object size. No such low-light box-level study was found. DarkQA (2025/26) is synthetic VQA and EgoNight (ICLR 2026) is video QA, neither with boxes.
- **Method part:**
  1. A VLM lists the classes present in each view; a reliability-weighted vote across views gives a class-presence prior.
  2. That prior (a) chooses the prompts for the open-vocabulary detector and (b) rescales the raw detector's box scores.
  3. Distil the prior into an image-level head on the raw detector, so no VLM is needed at inference.

**Closest prior work:** "Language Prompt vs. Image Enhancement" (arXiv 2604.10637, 2026), GEFU/SCUF (ICCV 2025, caption consistency for enhancement), VLOD-TTA (2025), and a 2024 challenge entry that fuses predictions over several enhanced views. The caption/class-presence consensus across enhanced views was **not found**.

**Target:** ACM MM 2027 (A\*, multimodal) or NeurIPS 2027 Evaluations & Datasets.

**Risks:** VLMs hallucinate in the dark, and ExDark's 12 closed classes leave text little to add. Use open-vocabulary or rare-class settings. **If the pilot is negative, drop the method part and fold the zero-shot results into Paper A.**

---

### Paper D (optional): *No-Regret Detection-Preserving Enhancement*, i.e. DPI-Det redesigned
**Type:** method with an image output. **Target:** Q1 journal: TCSVT, PR, TMM.

- **Keep:** the bounded multiplicative map, the frozen-detector feedback, and the human-viewable output.
- **Drop:** the Poisson-Gaussian SNR model, L_ret, the unsupervised U, the DCT loss, and feature consistency against raw (the identity trap).

**Method:**
1. Learn a harm map H, supervised by per-region and per-scale differences in detection loss between raw and an enhanced reference, under a frozen detector. This can reuse Paper B's utility signal.
2. Output I·(1 + (1−H)·A).
3. Add a hinge constraint L_det(I_out) ≤ L_det(I_raw) + ε, plus non-reference visibility losses so the image is useful to a human.

**Evaluation:** a Pareto curve of perceptual quality (NIQE, MUSIQ, a small user study) vs mAP under frozen detectors.

**Claim:** one image that is bright enough for a human operator and *never worse* for the machine (surveillance and driving use cases). **Novelty is about 5/10 (UNVERIFIED).** Must position against Sun et al. (NeurIPS 2022), FoCo (CVPR 2025), AMIEOD and TaskGuard.

---

### How the papers fit together
- **A** measures the enhancement-utility problem and explains it.
- **B** exploits utility per object at training time, with a raw detector at test time.
- **C** extends the question to language and VLMs.
- **D** produces an image that respects utility, for human viewing.

Each has a distinct contribution and distinct experiments. Section 2's finding that "harm is enhancer-determined, size-dependent and mostly domain shift" is the shared motivation.

---

## 7. Timeline (deadlines are estimates from 2026 cycles; confirm on the official sites)

| When | Work | Deadline |
|---|---|---|
| **Nov 3 – Nov 23, 2026** | Fix SCI. Add identity and brightness-matched controls. Run 3 seeds for YOLOv8n, RT-DETR and Faster R-CNN. Dump per-image predictions. Add the MAET 80/20 split. Run **P0** (zero-shot COCO detectors, no training). Start the Paper C pilot (inference only, in the background). | CVPR 2027 (Nov 16, 2026): **skip** |
| Nov 24 – Jan 31 | Paper A experiments: P0–P4 on ExDark + DarkFace + one more dataset; interventions; CKA; scaling; image-quality correlation; oracle headroom. Paper B prototype on YOLOv8n with the existing P3 teachers. | IJCAI 2027 (~mid Jan): skip |
| Feb 2027 | Write Paper A and **submit to a Q1 journal** (Pattern Recognition / TCSVT). Decide on Paper B: if +1.5 mAP over the best baseline on 2 datasets by mid-February, aim for ICCV. | ICCV 2027 (~Mar 1–8): B only if ready |
| Mar – Apr 1 | Paper B full baselines and ablations, or Paper C if its pilot was positive. | **ACM MM 2027 (~Apr 1)**: B or C |
| Apr – early May | Paper C benchmark with a released toolkit. | **NeurIPS 2027 E&D (~early May)**: C |
| May – Jul | Paper D. Revise Paper B, or resubmit after rejection. Write the thesis. | **AAAI-28 (~late Jul)**: B (primary or fallback) |
| Fallbacks | WACV / BMVC (CORE A, not A\*); Q1 journal versions of B or C | — |

---

## 8. Suggested thesis objectives (wording)

- **Objective 1:** *To systematically characterise when and why low-light image enhancement helps or harms object detection, across training protocols, detector families, object scales, illumination conditions and datasets, and to separate recoverable domain shift from irreversible information loss.* → Paper A
- **Objective 2:** *To design detection-preserving low-light learning methods that transfer the benefit of enhancement to the detector only where it helps (object- and scale-selective), without added inference cost, and to evaluate their cross-dataset generalisation.* → Paper B, with Paper D or C as extensions

---

## 9. Checklist for the next 2 weeks
- [ ] Re-run SCI from the official repo. Assert `output ≥ input` pixel-wise on 20 images. Regenerate SCI outputs and re-run P2.
- [ ] Add the identity control (raw through the same I/O path) and the brightness-matched gamma control.
- [ ] Train 3 seeds of YOLOv8n, RT-DETR and Faster R-CNN on raw ExDark. Save per-image predictions (JSON) for every test run.
- [ ] Write the bootstrap CI and paired-test script, plus Friedman / critical-difference plots.
- [ ] Re-run on the MAET 80/20 ExDark split.
- [ ] P0: COCO-pretrained YOLOv8n, RT-DETR and Faster R-CNN, zero-shot on raw and on 7 enhanced versions, with the 12 classes mapped to COCO.
- [ ] Start downloading DarkFace and LLVIP-visible.
- [ ] Paper C pilot: one VLM × 8 views × 737 images, measuring class-presence F1 and hallucination rate.
- [ ] Re-verify every reference by hand. Remove `utm_source=chatgpt.com` links and fix the LLVIP DOI.

---

### Files in this folder
- `analysis/reanalysis_of_reported_numbers.py` and `analysis/reanalysis_output.txt`: Section 2's statistics.
- `literature/01_benchmark_prior_art.md`: prior empirical and benchmark studies, with gaps.
- `literature/02_method_prior_art_dpidet_audit.md`: 27 related methods, ExDark/DarkFace state-of-the-art numbers, DPI-Det component audit.
- `literature/03_alternative_directions_vlm_distillation.md`: VLM/open-vocabulary, multi-view test-time augmentation, size-aware fusion, distillation, test-time adaptation, when-to-enhance, diffusion, low-light VLM benchmarks.
- `literature/04_reviewer_audit_and_venues.md`: area-chair-style audit, CORE rankings, deadlines, Q1 journals, experiment checklists, GPU-hour estimates.
