# Full-text novelty check: Paper 1 (study) and the no-regret enhancement variant (folded into Papers 1 and 2)

Checked 2026-11-03 against full texts (arXiv, CVF, NeurIPS proceedings). Each entry paraphrases the paper's own tables or text.

Not reachable, so marked **UNVERIFIED**: OpenReview review text (bot challenge), Kumari et al. (ScienceDirect 403).

## 1. What each work covers

● done, ◐ partial, blank = not done.

| Work | a | b | c | d | e | f | g | h | What it actually did |
|---|---|---|---|---|---|---|---|---|---|
| Wu et al., [arXiv 2409.14461](https://arxiv.org/abs/2409.14461) (preprint) | ◐ | | | ◐ | | | ◐ | | <ul><li>12 enhancers on ExDark (official split 3000/1800/2563) with RetinaNet, EfficientDet and YOLOv7, **trained and tested on enhanced images (P3 only)**. Most enhancers hurt, e.g. YOLOv7 76.09 raw vs 70.2–75.9.</li><li>For classification on synthetic Caltech256-LL, a clean-trained model gains from enhancement (VGG 0.74→0.81 with UTV), but a dark-trained model does not. This is already a **protocol-dependent sign flip**.</li><li>IQA–mAP Pearson (no CIs); t-SNE; says informally LLIE "cannot add new information". Single run; no size/condition analysis.</li></ul> |
| LIME-Eval, [arXiv 2410.08810](https://arxiv.org/abs/2410.08810) | ◐ | ◐ | | ◐ | | | ◐ | | <ul><li>YOLOX-m on ExDark under three protocols:<ul><li>Enhance-first P3: 45.9–47.6</li><li>Augment-first P3: 48.5–49.2</li><li>Raw: 49.2</li><li>COCO zero-shot (P0): raw 35.0 vs 34.0–35.4</li></ul></li><li>Quantization control ("Bread-round"); invokes the information bottleneck; detectors "more sensitive to noise and artifact" than colour/illumination.</li><li>One detector, 4 enhancers, no seeds or CIs.</li></ul> |
| Li et al., TPAMI 2022 survey, [arXiv 2104.10729](https://arxiv.org/abs/2104.10729) | ◐ | | | | | | | | <ul><li>Frozen DSFD (WIDER FACE) on 500 DarkFace images (P0). **Every** deep enhancer helps: AP50 0.195 raw vs up to 0.268.</li></ul> |
| Sun et al., [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/1cac8326ce3fbe79171db9754211530c-Paper 3onference.pdf) | ◐ | | | ◐ | | | | | <ul><li>YOLOv3 trained on clean VOC, tested on ExDark (P0-like): raw 46.27, SCI 48.47, ZeroDCE 41.01. **The sign depends on the enhancer.**</li><li>Covariate-shift argument; Eq. 5 constrained form min ‖x̂−x̃‖ s.t. ℓ(D(x̃),ŷ) ≤ r (needs clean x̂).</li></ul> |
| Cheng et al., [CVPRW 2026 LoViF](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/papers/Cheng_Generative_Low-Light_Enhancement_for_Computer_Vision_Revisiting_the_Gap_Between_CVPRW_2026_paper.pdf) (DBDiff) | ◐ | | | | | | ◐ | | <ul><li>Frozen COCO YOLOv8 (P0), 14 enhancers.</li><li>**ExDark: raw 0.565; 11 of 14 enhancers are lower** (RetinexNet 0.424). **DarkFace: raw 0.207; all 14 are higher** (0.226–0.262).</li><li>Method: AWF fusion I_F = σ(W)·I_L + (1−σ(W))·I_E with a pseudo-label detection loss.</li></ul> |
| TaskGuard, [arXiv 2609.08011](https://arxiv.org/abs/2609.08011) | | | | | | | | ◐ | <ul><li>**No low-light content** (zero mentions).</li><li>Synthetic COCO degradations plus DAWN rain; frozen YOLO11s/RT-DETR.</li><li>Exact regional counterfactual B_r = L_det(I_d) − L_det(I_d + M_r⊙R): 51.7% of regions positive, 48.3% negative.</li><li>A per-image restore-or-preserve policy.</li></ul> |
| Restore What Matters, [arXiv 2609.13791](https://arxiv.org/abs/2609.13791) | ◐ | | | | | | | ◐ | <ul><li>Turbulence face recognition (BRIAR); not low light, not detection.</li><li>Restoration-as-augmentation × test-time restoration; a restore-or-skip gate; PSNR-vs-recognition λ sweep; fine-tunes *restorer* layers, not the detector.</li></ul> |
| PreActIR, [arXiv 2610.20384](https://arxiv.org/abs/2610.20384) | | | | | | | | | Predicts IQA outcomes for agentic restoration. **No detection**; irrelevant. |
| NOD, [arXiv 2110.10364](https://arxiv.org/abs/2110.10364) | ◐ | | | | | ◐ | | | <ul><li>COCO-pretrained vs fine-tuned RetinaNet; 4 enhancers (P3) split by instance-level **extreme vs non-extreme** lighting; t-SNE.</li></ul> |
| Kumari et al., ASOC 2026, [S1568494626010550](https://www.sciencedirect.com/science/article/abs/pii/S1568494626010550) | ? | | | | | | | | **UNVERIFIED** (403). Snippets: YOLOv8–v11 on ExDark, DarkFace and LLVIP as a baseline. No evidence of an enhancer grid. |
| LR-YOLO, [arXiv 2510.24232](https://arxiv.org/abs/2510.24232) (NeurIPS 2025) | | | | ◐ | ◐ | | | | <ul><li>Lipschitz/Jacobian mechanism: the detector's Jacobian norm is about 10× the restorer's.</li><li>On ExDark, cascades hurt: YOLOv8 50.0 vs LLFormer 46.6.</li><li>On synthetic COCO_Dark, InstructIR→YOLOv8 gives **AP_S −1.5 but AP_L +1.4**. This is the *opposite* size pattern to our.</li></ul> |
| MPD ("From Human to Machine Preference"), [CVPR 2025](https://arxiv.org/abs/2503.10078) | | | | | | | ◐ | | Synthetic distortions, including darkening, with frozen models. Human-centric IQA fails to predict machine preference. No LLIE outputs. |
| Pei et al., [ECCV 2018](https://arxiv.org/abs/1810.05716) (haze, classification) | ◐ | | | ◐ | | | ◐ | | <ul><li>Clean-train/test-dehazed helps (21.7→42.5%); hazy→hazy ≈ dehazed→dehazed. PSNR/SSIM rank correlation; feature reconstruction.</li><li>**No training-set-size study.**</li></ul> |
| *Is UIE All Detectors Need?* [arXiv 2311.18814](https://arxiv.org/abs/2311.18814) (underwater; journal venue UNVERIFIED) | ◐ | | ◐ | ◐ | | | ◐ | | <ul><li>18 enhancers × 7 detectors, P3 only. Raw is best for every detector.</li><li>**Stronger backbones and longer schedules widen the raw–enhanced gap.**</li><li>TIDE, CAM, observational attribution to colour/noise/edges.</li></ul> |
| *Revisiting UIE for OD*, [J. Imaging 2026](https://doi.org/10.3390/jimaging12010018) (underwater) | | | | | | ◐ | ◐ | ● | <ul><li>Per-image mAP and a **"mixed-set upper bound"**: the best of {original, 9 enhancers} for each image.</li><li>Quality-stratified effect: enhancement helps low-quality images and over-enhances high-quality ones.</li></ul> |
| AMIEOD, [arXiv 2605.06084](https://arxiv.org/abs/2605.06084) (TMM) | ◐ | | | | | | | ◐ | <ul><li>YOLOv3 on ExDark (MAET split):<ul><li>Raw: 76.5</li><li>SCI-trained (P3): 76.6</li><li>Mixed raw+SCI (P4): 76.9 on raw, 76.2 on SCI</li></ul></li><li>Argmin-loss labels over {raw, experts}; no oracle mAP.</li></ul> |
| FoCo, [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Gu_Improving_Visual_and_Downstream_Performance_of_Low-Light_Enhancer_with_Vision_CVPR_2025_paper.pdf) | ◐ | | | | | | | | P0-only evaluation; enhancer trained with frozen-foundation-model task losses. |
| VRD-IR, [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Visual_Recognition-Driven_Image_Restoration_for_Multiple_Degradation_With_Intrinsic_Semantics_CVPR_2023_paper.pdf) | | | | | | | | | Recognition-driven restoration with frozen models. Relevant to D only. |

## 2. Status of each angle

- **(a) Protocol matrix and sign flip: PARTIAL, and the headline is at risk.**
  - Already shown piecewise: Pei 2018 and 2409.14461 (classification); LIME-Eval (3 protocols, 1 detector); P0 on DarkFace helps (survey, LoViF); P3 on ExDark hurts (2409.14461, LIME-Eval); P4 (AMIEOD, one enhancer).
  - **Two independent full-text P0 results on ExDark (LIME-Eval, LoViF) show that enhancement does *not* help frozen COCO detectors there.** The roadmap's bet that "P0 is where enhancement helps" will probably fail on ExDark and succeed on DarkFace.
  - Still open: the same enhancers × detector families under P0–P4 on ≥3 real datasets with seeds and CIs. The defensible headline is a **protocol × darkness interaction**, not "the sign flips with protocol".
- **(b) Shift vs irreversible decomposition: OPEN as an operational decomposition**, because nobody reports P2 alongside P3 for detection. The DPI/information intuition itself is folklore (2409.14461's introduction; LIME-Eval's information-bottleneck remark). Two problems need fixing:
  1. **LIME-Eval's augment-first P3 (48.5–49.2) ≈ raw (49.2)**, while enhance-first loses 1.6–3.3 mAP. This suggests much of P1−P3 may be an artefact of augmentation order: Mosaic, mixup or colour jitter applied to pre-enhanced images. Ultralytics applies these by default. Run both orders, an identity-I/O control and a quantization control, or reviewers will dismiss the "17–42% irreversible" numbers.
  2. The DPI bounds Shannon information, not *usable* information. A finite-data, finite-capacity detector can do better *or* worse after processing. Frame P1−P3 as the **residual (non-adaptable) penalty** and cite V-information ([Xu et al. ICLR 2020, arXiv 2002.10689](https://arxiv.org/abs/2002.10689)). Do not call it information loss.
- **(c) Training-set size and frozen vs fine-tuned backbone: OPEN.**
  - Targeted searches (data size, frozen vs fine-tuned, linear probe, few-shot) found no match in low light, haze (Pei 2018 uses fixed sizes), underwater or super-resolution.
  - Nearest analogues to cite: the underwater study (gap *widens* with backbone capacity); Restore What Matters (fine-tunes restorer layers).
- **(d) Controlled interventions plus CKA/MMD regressed on per-object ΔAP: OPEN.** Searches for CKA and for gain/colour/noise-component ablations found nothing for LLIE. Cite as partial precedents: LIME-Eval's synthetic sensitivity test, the underwater study's observational attribution, LR-YOLO's Jacobian analysis, Sun's feature shift, and t-SNE plots (2409.14461, NOD, ExDark).
- **(e) Object size across detectors: PARTIAL.** Only synthetic COCO-d (LDWLE: small objects gain; snippet-level, not re-read) and COCO_Dark (LR-YOLO: small objects *lose*, large *gain*). The real-data, multi-detector, CI-backed result is open, and contradicting LR-YOLO's direction is a selling point.
- **(f) ExDark's 10 lighting conditions: OPEN for detection.** Partial precedents:
  - NOD: 2 conditions, 4 enhancers, 1 detector.
  - J. Imaging 2026: quality-stratified, underwater.
- **(g) IQA vs ΔmAP with bootstrap CIs: PARTIAL, effectively DONE except the CIs.** Covered by 2409.14461, LIME-Eval, Pei 2018, MPD, the underwater study and LoViF. Treat this as hygiene, not a contribution.
- **(h) Per-image oracle headroom: PARTIAL.** The concept is done (J. Imaging 2026's mixed-set upper bound; AMIEOD's argmin-loss labels; TaskGuard's counterfactual utility). A low-light quantification against TaskGuard/AMIEOD-style selectors is new but minor.

## 3. What Paper 1 can and cannot claim

**Safe to claim as novel** (wording to adopt once executed, seeds and CIs included):
1. "The first low-light detection study to evaluate the same enhancers and detector families under all of zero-shot, raw-trained, enhanced-trained and mixed protocols on ≥3 real datasets with paired-bootstrap CIs. It shows that the sign of the enhancement effect is governed by the interaction of protocol and scene darkness."
2. "An operational decomposition of the enhancement penalty into an adaptable component (P3−P2) and a residual component (P1−P3), controlled for augmentation order, quantization and I/O, and stratified by size, lighting condition and detector family."
3. "The first measurement of how enhancement utility scales with training-set size and backbone freezing in low-light detection."
4. "A controlled-intervention decomposition of enhancer outputs (luminance-matched gain/gamma, colour, noise residual, detail) linked to layer-wise representation shift (CKA/MMD) and per-object ΔAP."
5. "Real-data, cross-detector evidence on object size (contrary to the synthetic COCO-Dark trend in LR-YOLO) and on ExDark's 10 lighting conditions."

**Must NOT write:**
- "We are the first to show that low-light enhancement does not help / harms detection." (UG²+, Sun 2022, 2409.14461, LIME-Eval, LoViF.)
- "We are the first to show that retraining on enhanced images fails to beat raw images." (2409.14461, LIME-Eval.)
- "We are the first to evaluate enhancers with frozen, normal-light-pretrained detectors." (Survey, Sun 2022, LIME-Eval, FoCo, LoViF.)
- "We are the first to show that the effect of enhancement depends on the training protocol." (Pei 2018, 2409.14461, LIME-Eval.)
- "Enhancement improves zero-shot COCO detectors on ExDark." Contradicted by LIME-Eval and LoViF; claim it only if your CIs show it.
- "Image-quality metrics do not predict detection utility" as a finding. (Pei 2018, 2409.14461, LIME-Eval, MPD, underwater.)
- "By the data-processing inequality, P1−P3 quantifies irreversible information loss." This is technically wrong: usable information is not bounded by the DPI, and augmentation order is a confound.
- "We introduce a per-image oracle / selective-enhancement upper bound." (J. Imaging 2026, TaskGuard, AMIEOD.)
- "First TIDE analysis" or "stronger detectors do not remove the penalty." (Underwater study.)
- "Small objects benefit from enhancement" as a first. (LDWLE; LR-YOLO shows the opposite on synthetic data.)

## 4. Verdict on the no-regret enhancement variant (not a standalone paper)

**Novelty: 3/10 as written** (down from the earlier unverified 5/10). Every component has a full-text precedent:

| Component | Precedent |
|---|---|
| Harm map supervised by per-region raw-vs-enhanced detection-loss difference | Exactly TaskGuard's B_r^CF. AMIEOD's per-image argmin label. |
| Spatial raw/enhanced mixing map | DBDiff's AWF; FCMA-Det (TGRS 2025, per AMIEOD's description; not read) |
| Constraint on detection loss | Sun 2022, Eq. 5, absolute bound |
| Frozen-detector guidance | Sun 2022, FoCo, VRD-IR, DBDiff's TAM |
| Human/machine Pareto evaluation | Restore What Matters λ sweep; [UniRestore](https://arxiv.org/abs/2501.13134), CVPR 2025 (task-specific prompts, i.e. *different* outputs per consumer) |

The only thin gap: no low-light paper imposes an explicit *relative-to-raw* constraint.

**Sharpest defensible formulation:**

*Non-inferiority-constrained enhancement.* Maximise a no-reference visibility objective subject to, for every image i and object-scale band s:

E[L_det(f(I_i))_s − L_det(I_i)_s] ≤ ε

Solve it with a primal-dual (Lagrangian) method rather than a fixed-weight hinge. The identity map is always feasible.

- **Headline metric: regret rate.** The fraction of images or objects whose AP or loss is worse than raw, with CIs, measured on **held-out detectors and datasets**.
- **Show the selectors are special cases.** TaskGuard and AMIEOD are binary, per-image versions of this constraint (H ∈ {0,1}). Show Pareto dominance over DBDiff, FoCo, TaskGuard-gated enhancers, AMIEOD-style selection and Paper 1's per-image oracle.

**Recommendation:** do not run D as a standalone paper.
- Fold the regret-rate metric and a simple gated baseline into **Paper 1, section (h)**.
- Fold the harm map into **Paper 2** as the per-object utility signal.
- Spin D out only if the cross-detector regret rate is near 0 and D Pareto-dominates DBDiff. That version would score about 5/10, and it is a TCSVT-level paper at best.

## 5. LIME-Eval's OpenReview record (ICLR 2025, withdrawn)

**Verified** from the papercopilot ICLR 2025 dump (`iclr/iclr2025.json`, id `hYU0P4Wlj9`):

| Field | Values |
|---|---|
| Ratings | 3 / 6 / 8 / 8 |
| Confidence | 5 / 4 / 4 / 4 |
| Soundness | 1 / 2 / 3 / 3 |
| Contribution | 1 / 3 / 3 / 3 |

Withdrawn despite a mean rating of 6.25, with no author replies. The only confidence-5 reviewer scored soundness 1 and contribution 1.

**Review text: UNVERIFIED.** OpenReview requires a bot challenge, which I did not bypass.

The weaknesses below are **inferred from the full text**, not quoted from reviewers. Paper 1 should pre-empt all of them:
1. **One detector and one dataset for the protocol claim** (YOLOX-m on ExDark), with differences of about 0.4 mAP and no seeds or CIs.
2. **Unequal tuning.** Table 1 was "carefully tuned", while the other experiments used authors' defaults.
3. **The order of enhancement and augmentation flips the conclusion.** Report both orders, or the protocol critique cuts against you.
4. **Small correlation samples:** the human–detector Spearman ρ = 0.70 rests on 15 points.
5. **Weak metric validity:** confidence energy is not accuracy under shift; it was calibrated on synthetic COCO; the correlation "vanishes" for YOLOX-Tiny.
6. **The user study does not measure machine utility,** and ties were handled ad hoc.

**Paper 1 must include:** ≥3 detector families and ≥3 real datasets; 3 seeds with paired-bootstrap CIs; identical tuning budgets across arms; both augmentation orders; identity and quantization controls; a capacity sweep; correlations over ≥50 cells with CIs.
