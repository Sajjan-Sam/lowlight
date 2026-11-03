# Method prior-art audit (incl. the earlier DPI-Det design) (as of 2026-11-03)

**How this was checked.** arxiv.org, CVF, OpenReview and the publisher sites could not be reached from this session (network policy, 403). Each entry was checked against WebSearch snippets taken from the primary pages (arXiv, CVF, NeurIPS, AAAI, ICLR) and against the authors' GitHub READMEs, which I read directly. The search budget ran out before I could pull the full results tables. **[S]** means the number comes from a snippet of the paper. **[GH]** means it comes from the authors' repository. **UNVERIFIED** means it comes from a secondary source or could not be confirmed.

**Splits.** ExDark numbers are not comparable across protocols. Many YOLOv3 papers follow MAET with an 80/20 split, about 1,473 test images; the EMV-YOLO README confirms 80/20. The YOLA and FRBNet protocol uses different baselines. Our 70/20/10 split (737 test images) matches none of these. Trash-to-Treasure reportedly used 737 test images, but only a snippet supports that (UNVERIFIED).

## A. Closest methods

| # | Method (venue) | Link | Core idea | Data | Reported numbers |
|---|---|---|---|---|---|
| 1 | Dynamic LLIE for OD (ICPR 2020) | doi 10.1109/ICPR48806.2021.9412802 | Selects enhancement sub-networks per sample, trained end-to-end with detection | UNVERIFIED | UNVERIFIED |
| 2 | YOLO in the Dark (ECCV 2020) | ecva.net 3723 | Glue layers merge SID and YOLO; generator distilled from both models | SID | – |
| 3 | MAET (ICCV 2021) | arXiv 2205.03346 | Physical noise model plus ISP degradation; multitask AET; orthogonal regularity | ExDark, DarkFace | ExDark 77.7 against a YOLOv3 baseline of 76.4 (renewed table); DarkFace 56.2 [GH] |
| 4 | Crafting OD in Very Low Light (BMVC 2021) | arXiv 2110.10364 / LODDataset | Unprocess-plus-noise synthesis; auxiliary recovery branch | LOD (RAW) | – |
| 5 | IA-YOLO (AAAI 2022) | arXiv 2112.08088 | CNN-PP predicts DIP filter parameters, trained by detection loss | VOC-dark, ExDark | 77.8 (as reproduced by AMIEOD) [S] |
| 6 | DENet / DE-YOLO (ACCV 2022) | CVF ACCV2022 | Laplacian-pyramid enhancer of about 45K params, detection-driven | ExDark | 77.3 [GH, MAET table] |
| 7 | GDIP (ICRA 2023) | arXiv 2209.14922 | Gated weighted mix of differentiable IP ops, trained by detection loss | ExDark | Different protocol; YOLOv3 baseline 37.03 [S] |
| 8 | Rethinking IR for OD (NeurIPS 2022) | NeurIPS 2022 | Targeted adversarial attack on a frozen detector gives detector-friendly pseudo-GT that stays close to the clean image | Dehaze, LLIE | – |
| 9 | VRD-IR (CVPR 2023) | CVF CVPR2023 | Recognition-driven restoration in a semantic feature space | Cls, det, ReID | – |
| 10 | SKF (CVPR 2023) | arXiv 2304.07039 | Segmentation prior gives region-wise LLIE losses | LOL | – |
| 11 | PE-YOLO (ICANN 2023) | arXiv 2307.10953 | Pyramid enhancement plus YOLOv3, detection loss only | ExDark | 78.0 [S] |
| 12 | FeatEnHancer (ICCV 2023) | arXiv 2308.03594 | Hierarchical feature-level enhancement driven by task loss | ExDark, DarkFace | +5.7 on ExDark; 86.3 ExDark and 69.0 DarkFace with Featurized Query R-CNN [GH] |
| 13 | 2PCNet (CVPR 2023) | arXiv 2303.13853 | Day-to-night UDA; teacher-student; NightAug | BDD100K | – |
| 14 | Similarity Min-Max (ICCV 2023) | arXiv 2307.08779 | Darkening plus feature-similarity maximisation, zero-shot | Multiple tasks | – |
| 15 | RAW OD / ROD (CVPR 2023) | CVF CVPR2023 | Image-adaptive HDR RAW adjustment, trained end-to-end | ROD | – |
| 16 | Trash to Treasure (AAAI 2024) | arXiv 2309.03548 | Retinex decomposition; illumination fed to the detector as auxiliary input | ExDark | UNVERIFIED |
| 17 | DAI-Net (CVPR 2024) | arXiv 2312.01220 | Frozen Retinex net; reflectance-representation learning; zero-shot day-to-night | DarkFace, ExDark | DarkFace 28.0 zero-shot / 52.9 supervised (DSFD) [GH]; ExDark 78.3 UNVERIFIED |
| 18 | YOLA (NeurIPS 2024) | arXiv 2410.18398 | Lambertian illumination-invariant kernels learned with the detector | ExDark, DarkFace | ExDark 72.7 (YOLOv3) / 75.3 (TOOD); DarkFace 61.5 / 67.5 [GH] |
| 19 | RAW-Adapter (ECCV 2024) | arXiv 2408.14802 | Learnable-ISP input adapters plus model-level adapters into the backbone | LOD, PASCAL-RAW | – |
| 20 | EMV-YOLO (BMVC 2024) | BMVC 2024 paper 262 | 27K-param Retinex module that balances R and L for the detector | ExDark, DarkFace | 79.7 (per AMIEOD) [S] |
| 21 | FoCo (CVPR 2025) | CVF CVPR2025 | LLIE trained with feature and task alignment from foundation models | LLIE, det, seg | – |
| 22 | Dark-ISP (ICCV 2025) | arXiv 2509.09183 | Differentiable linear and nonlinear ISP tuned by detection loss | 3 RAW sets | – |
| 23 | FRBNet (NeurIPS 2025) | arXiv 2510.23444 | Frequency-domain radial-basis filter on an extended Lambertian model | ExDark, DarkFace | ExDark 74.9 vs 71.0 base (YOLOv3), 75.4 vs 72.5 (TOOD); DarkFace 57.7 / 65.1 [S] |
| 24 | LR-YOLO (NeurIPS 2025) | arXiv 2510.24232 | Lipschitz regularisation for restoration-then-detection instability | Haze, low light | – |
| 25 | SimROD (AAAI 2026) | arXiv 2503.07101 | 4-parameter global gamma plus green-guided local enhancement | ROD, LOD | Beats RAW-Adapter [S] |
| 26 | SGLDet (ICLR 2026) | proceedings.iclr.cc 2026 | Training-only enhancement, denoising and Fourier fusion; zero extra inference cost | ExDark, DarkFace | DarkFace 76.6 (+16.9 over DSFD), UNVERIFIED |
| 27 | AMIEOD (arXiv 2605.06084, 2026) | arXiv 2605.06084 | Multi-expert enhancers; expert selection trained with detection-guided cross-entropy; regression toward the expert with the lowest detection loss | ExDark, DarkFace, LLVIP, M3FD | ExDark 82.1 vs 76.5 base (YOLOv3); best on DarkFace (+0.1) and LLVIP (+0.4) [S] |

Also relevant:
- Clean Feature Distillation (PRCV 2025): clean teacher distilled into a low-light student.
- Feature Corrective Transfer Learning (CVPRW 2024): feature-discrepancy loss between ideal and non-ideal inputs.
- WTEFNet (arXiv 2505.23201): wavelet transform plus an illumination-confidence soft mask.
- Unsupervised Illumination Adaptation (TPAMI 2024): an enhancement curve designed for machine vision.
- VLOD-TTA (2025): test-time adaptation that improves vision-language detectors on low-light data.
- CVPRW 2026 (LoViF): "Generative LLIE for CV: Revisiting the Gap Between Perceptual Quality and Detection Utility".
- Empirical study arXiv 2409.14461.
- Not checked in detail (UNVERIFIED): IAFE-YOLO+ (Pattern Recognition 2026), IIF-Det (ACM 2025), Dark-YOLO (Appl. Sci. 2025; 71.3 mAP50).
- REDI: I could not identify this work (UNVERIFIED).

**Current best numbers by protocol.**
- **ExDark, MAET protocol (YOLOv3):** about 82 (AMIEOD), against a baseline of about 76.4–76.5.
- **ExDark, YOLA/FRBNet protocol:** about 75.4 (TOOD).
- **DarkFace:** about 65–69 under supervised YOLOv3/TOOD/FQ-RCNN, and 52.9 under DSFD on the CodaLab test set. SGLDet's 76.6 is UNVERIFIED.
- **LLVIP-visible:** no clean state-of-the-art number found.

Our YOLOv8n result (73.3) cannot be compared with any of these.

## B. Component-by-component overlap

| Component | Closest prior work | Novelty (1–10) |
|---|---|---|
| **Retinex-SNR encoder (R, L, N, S = μ/√(αμ+β))** | Retinex fed to a detector: DAI-Net, Trash-to-Treasure, EMV-YOLO. Lambertian invariance: YOLA, FRBNet. SNR map: SNR-Aware (CVPR 2022). Physical noise models: MAET, BMVC 2021, ELD. I checked the SNR-Aware code: its SNR map is a heuristic, blur/abs(I−blur), computed on grey sRGB, not Poisson-Gaussian. | **2** |
| **Bounded multiplicative map W = 1 + A** | Zero-DCE bounded pixel-wise curves; SCI/RUAS I/L; IA-YOLO and GDIP gain/gamma; SimROD global gamma; Dark-ISP gains; SACC. "Stay close to the input" comes from Rethinking-IR-for-OD. | **2** |
| **Object mask (GT boxes at train time, proposals at test time)** | Box-masked foreground/background feature losses are standard in detector distillation (FGD CVPR 2022, fine-grained imitation CVPR 2019; recalled from memory, not re-checked). SKF uses region priors. I found no exact "box-masked enhancement for detection" (absence UNVERIFIED). | **4** |
| **Uncertainty gate U** | Image-level gating or selection: GDIP, ICPR 2020, AMIEOD. Illumination-confidence soft mask: WTEFNet. Uncertainty maps in LLIE: GSAD, U2CLLIE. As written, U has no supervision signal. | **3** |
| **FiLM adapter conditioned on Retinex maps** | RAW-Adapter's model-level adapters, Trash-to-Treasure (illumination into the detector), EMV-YOLO, YOLA. I found no FiLM-specific instance (UNVERIFIED), but conditioning a backbone on illumination is established. | **3** |
| **Masked feature-consistency loss (raw vs DPI)** | SMM, DAI-Net, FCTL (CVPRW 2024), FoCo, Clean-Feature-KD, AMIEOD's DGRL. Logical flaw: matching DPI features to *raw* features under a detector trained on raw images rewards W ≈ 1, i.e. the identity. | **2** |
| **DCT frequency loss** | Focal Frequency Loss (from memory), DEFormer (DCT), FRBNet, SGLDet's Fourier fusion, PE-YOLO and DENet (Laplacian), WTEFNet (wavelet). | **1–2** |
| **Freeze detector, train the module with detection loss, then unfreeze neck/head** | IA-YOLO, AMIEOD (two-stage scripts), Rethinking-IR (frozen detector), FoCo, ICPR 2020. | **2** |
| **The whole combination** | A kitchen-sink union of 8 losses and 5 modules. | **3** |

## C. Verdict

**Would DPI-Det be accepted at CVPR, ICCV or ECCV as written? No; it would very likely be rejected.** Every component already exists in prior work from 2020–2026. The adaptive, detection-driven enhancement story is already taken by ICPR 2020, IA-YOLO/GDIP and AMIEOD. The feature-level and illumination-invariant line is mature: FeatEnHancer, YOLA, FRBNet, SGLDet. Nothing has been run yet. The current evidence is single-seed YOLOv8n on a non-standard split. Realistically this is a mid-tier conference or journal contribution unless it is re-centred.

**Sharpest defensible idea to carve out: a "no-regret", harm-aware enhancement.** Our own data is the hook. Every enhancer hurts raw-trained detectors, even after retraining (P3). The harm depends on scale: small objects are sometimes helped, while medium and large objects are consistently hurt. The method would be:
- Learn a spatial **harm map** H(x) that predicts where enhancement raises detection loss.
- Supervise H with per-region or per-scale detection-loss differences between raw and enhanced inputs under a frozen detector.
- Apply I·(1 + (1−H)·A), so the transform stays anchored to the raw image.
- Add an explicit hinge constraint, L_det(I_dpi) ≤ L_det(I_raw), at the image level and at each scale.

The claim would be enhancement that is guaranteed, by construction and checked empirically, never to underperform raw at the region and scale level. This differs from AMIEOD and ICPR 2020, which select per image, have no raw anchor and offer no guarantee. Novelty is about 5/10; it is plausible but I could not fully check it (UNVERIFIED) because the search budget ran out.

**Fallback.** A rigorous "why LLIE hurts detection" causal analysis would suit a Q1 journal (TIP or TCSVT). It would separate four factors through controlled interventions: noise amplification, BatchNorm/feature shift, contrast and colour shift on medium/large textures, and Lipschitz instability. That fits arXiv 2409.14461, LR-YOLO and the CVPRW 2026 paper, so it is not novel by itself.

**Baselines it must beat:**
- Protocols: the MAET 80/20 YOLOv3 protocol (base 76.4) and the YOLA/FRBNet TOOD protocol, plus DarkFace, LLVIP-visible and BDD-night.
- Methods: IA-YOLO, GDIP, DENet, PE-YOLO, MAET, FeatEnHancer, YOLA, DAI-Net, EMV-YOLO, FRBNet, SGLDet, AMIEOD and LR-YOLO.
- Controls: a raw detector fine-tuned for the same extra epochs; raw training with brightness, gamma and noise augmentation; and a SimROD-style global gamma.

**What reviewers will attack:**
1. **Poisson-Gaussian SNR on sRGB JPEG.** ExDark images are web-sourced, tone-mapped and compressed by unknown ISPs, so α and β cannot be identified. Either estimate a noise-level function in sRGB or drop the physics claim.
2. **GT boxes at train time vs proposals at test time.** This is a train/test mismatch and arguably label leakage. Proposals need a second pass. We must ablate with predicted masks during training too.
3. **Kitchen-sink design.** 8 losses and many λ, a and b hyperparameters create an ablation explosion. Gains will be attributed to extra capacity or extra fine-tuning.
4. **Trivial identity solution.** L_feat, L_id and a frozen raw-trained detector all push W toward 1. Our own P2/P3 results predict that raw is the optimum.
5. **Unclear supervision for L_ret and L_snr.** ExDark has no paired normal-light references, so it is not clear what supervises these losses.
6. **Inference cost.** Compare against EMV-YOLO (27K params), SimROD (4 params) and SGLDet (zero added inference cost).
7. **Weak evaluation.** Single seed, a custom split, and differences of about 1 mAP with no confidence intervals. The "LLIE hurts detection" finding is already known (Rethinking-IR, 2409.14461, LR-YOLO, CVPRW 2026).
8. **Overlap with AMIEOD and ICPR 2020.** The positioning has to be sharp against both.
