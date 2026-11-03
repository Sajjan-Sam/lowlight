# Narrower Claims Keep Both Papers Publishable

Neither paper has been scooped as of 3 November 2026, but they are not equally novel, and neither will set records on its own. **Paper 1, the Q1-journal study, is about 6/10 novel.** No published paper runs the same enhancers and detector families under all five protocols (P0–P4) on three or more real datasets with seeds and confidence intervals. None reports the raw-trained/enhanced-test protocol (P2) next to enhanced-trained (P3). None sweeps training-set size or backbone freezing, and none links controlled component interventions to representation shift and per-object ΔAP. Acceptance therefore rests on those last three pillars, and on fixing a confound reviewers will spot at once: darkness is entangled with dataset identity. **Paper 2, "Distilling Computable Privilege", is about 4.5/10 novel.** The full combination is unpublished, but cross-fitting inside distillation (Dao et al.), validation-gated per-sample distillation from a privileged teacher (DIW, July 2026) and loss-supervised per-location view routing (ERA-Det, October 2026) all exist. That leaves the raw-vs-raw seed-noise significance test as the one load-bearing ingredient no prior work has. On results, published numbers predict that Paper 1 yields a publishable finding under almost every outcome. Paper 2 should beat plain raw training by +2–4 mAP50, mostly through teacher capacity. Its gate, however, is expected to add only +0.3–1.0 over ungated distillation, with confidence intervals that may straddle zero on ExDark. That points Paper 2 to **ACM MM 2027 (deadline around early April) as the primary venue**, with ICCV 2027 (around 5–8 March) only if the go/no-go gates pass by about 10 February. Paper 1 should go to **IEEE TMM, with TCSVT as an equal alternative**, around late March to April 2027. The two plans below are complete and executable, but together they need **about 1,870 GPU-hours (about 1,780 after sharing overlapping runs)**. That does not fit on one 24 GB GPU inside Paper 2's 12-week window unless Paper 1 takes its lean path or gets a second GPU.

## Neither paper is scooped, but only Paper 1 has room to spare

Both re-checks searched arXiv through its API, sorted by date, reading every hit since May 2026 and the September–November 2026 listings line by line. They also searched Crossref, OpenAlex and the web for journals. They could not reach ScienceDirect, MDPI or IEEE Xplore early-access listings, so a journal-only paper from October–November 2026 that these indexes do not surface is the remaining scoop risk for both papers ([Paper 1 re-check](/home/user/lowlight/research_notes/Low%20light%20detection%20paper%20plans/paper1_novelty_recheck.md); [Paper 2 re-check](/home/user/lowlight/research_notes/Low%20light%20detection%20paper%20plans/paper2_novelty_recheck.md)).

### Paper 1 keeps four open gaps and loses only its "first observation" framing

Every published low-light study covers a single slice of the design. LIME-Eval runs three protocols on one detector and one dataset, with no seeds or CIs ([arXiv 2410.08810](https://arxiv.org/abs/2410.08810)). Wu et al. run 12 enhancers × 3 detectors on ExDark, P3 only, and remain an unreviewed preprint ([arXiv 2409.14461](https://arxiv.org/abs/2409.14461)). The CVPRW 2026 LoViF paper runs 14 enhancers through frozen detectors (P0 only) on ExDark and DarkFace ([CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/papers/Cheng_Generative_Low-Light_Enhancement_for_Computer_Vision_Revisiting_the_Gap_Between_CVPRW_2026_paper.pdf)). AMIEOD reports mixed training (P4) for one enhancer ([arXiv 2605.06084](https://arxiv.org/abs/2605.06084)). The closest analogue in another domain, the underwater "Is UIE All Detectors Need?" study in IEEE JOE 2024, trains 133 models but under P3 only ([DOI 10.1109/JOE.2023.3302888](https://doi.org/10.1109/JOE.2023.3302888)). New items from September–October 2026 add rigour precedents but scoop nothing: AegisBench (three detector families, three seeds, no enhancers) ([arXiv 2611.00077](https://arxiv.org/abs/2611.00077)), a component-wise sensor-degradation study on satellite imagery ([arXiv 2609.38265](https://arxiv.org/abs/2609.38265)), and a one-detector SPIE evaluation in which raw input scores highest ([DOI 10.1117/12.3125752](https://doi.org/10.1117/12.3125752)).

| # | Planned contribution | Verdict | Closest work | What keeps it open |
|---|---|---|---|---|
| 1 | Same enhancers × detector families under P0–P4 on ≥3 real datasets; protocol × darkness interaction | **Partial**: novel only as the integrated, CI-backed matrix | LIME-Eval, LoViF, Wu et al., AMIEOD, Kumari et al. ASOC 2026 (scope unread) | Nobody has P2, more than one protocol on more than one dataset, seeds or CIs. The ExDark-vs-DarkFace contrast is **already visible in LoViF Table 2**, so it is new only as a tested statistical claim |
| 2 | Adaptable penalty (P3 − P2) vs residual penalty (P1 − P3), controlled for augmentation order, quantisation and I/O | **Novel** (nobody reports P2 next to P3) | LIME-Eval's augmentation-order and quantisation controls | The concept is folklore; the operational decomposition is unreported. Risk: the residual may be ≈ 0 |
| 3 | Utility vs training-set size (10–100%) and frozen vs fine-tuned backbone | **Novel** (no precedent in low light, haze, underwater or super-resolution) | Underwater study: the raw–enhanced gap widens with capacity | Nobody sweeps data fraction or freezing with enhanced inputs |
| 4 | Gain/colour/noise/detail interventions → layer-wise CKA/MMD → per-object ΔAP | **Novel for LLIE**; methodology partial | Satellite component degradations; CKA under MRI artefacts ([arXiv 2608.06613](https://arxiv.org/abs/2608.06613)) | Nobody decomposes *enhancer outputs* and links them to representation shift and per-object ΔAP |
| 5 | Real-data size effect; ExDark's 10 lighting conditions | **Partial**, at higher risk than the roadmap assumed | LR-YOLO (synthetic); IAFE-YOLO+ (PR 2026) reports per-lighting-condition mAP for its own module ([DOI 10.1016/j.patcog.2026.113122](https://doi.org/10.1016/j.patcog.2026.113122)) | Claim only "first **enhancer × protocol** analysis across ExDark's 10 conditions" |

Two risks need action before submission. **The highest-risk unread item is Kumari, Singh and Saurabh's Applied Soft Computing 2026 review with benchmark experiments** (YOLOv8–v11 on ExDark, DarkFace and LLVIP). Its abstract could not be retrieved, so whether it crosses enhancers × detectors × protocols is unknown ([DOI 10.1016/j.asoc.2026.115607](https://doi.org/10.1016/j.asoc.2026.115607)). We must obtain the full text. The second risk is structural. **"Darkness" is confounded with dataset identity**: ExDark has 12 general classes under mixed lighting, DarkFace has one class of tiny faces, and NOD is driving scenes. A reviewer will say the interaction is task × protocol, not darkness × protocol. The plan below therefore tests the interaction *within* datasets as well: ExDark's lighting groups, NOD's per-box extreme-light flags, per-image luminance quartiles, and a new physics-based darkening ladder on well-lit COCO images. Without these, the headline is vulnerable.

The must-not-claim list from the earlier full-text audit still stands. Do not claim to be first to show that enhancement hurts detection, that retraining on enhanced images fails to beat raw, or that enhancers can be evaluated with frozen detectors. Nor that effects depend on protocol, that IQA metrics fail to predict utility, or that small objects benefit. Do not introduce a per-image oracle as new, and do not call P1 − P3 "information loss". The data-processing inequality bounds Shannon information, not usable information, so frame P1 − P3 with V-information ([arXiv 2002.10689](https://arxiv.org/abs/2002.10689); [literature audit 06](/home/user/lowlight/research/literature/06_paper1_study_novelty.md)). Also avoid the "systematic review/benchmark" framing, which Applied Soft Computing and KBS already hold and which invites "incremental" reviews.

### Paper 2's last unclaimed ingredient is the seed-noise null

The earlier audit had already removed per-instance, per-teacher, ground-truth-loss gating from the claims, because M-AID owns it ([M-AID PDF](https://par.nsf.gov/servlets/purl/10405166)). The November re-check strips C1 further.

| Ingredient | Closest prior work | What that work does | What still separates us |
|---|---|---|---|
| Cross-fitted teachers | Dao et al., *KD as Semiparametric Inference* ([arXiv 2104.09732](https://arxiv.org/abs/2104.09732)) | Out-of-fold teachers de-bias distillation *targets* | We use cross-fitting to **estimate a per-object treatment effect of a view**, not to de-bias targets; multiple teachers; detection |
| Held-out, per-sample gating of a privileged teacher | DIW, July 2026 ([arXiv 2608.24904](https://arxiv.org/abs/2608.24904)) | Four-IMU teacher → one-IMU student; fold-internal meta-set gates logit and feature losses per sample (macro-F1 0.5716 → 0.6385) | Single teacher, activity recognition, meta-gradient instead of a cross-fitted effect, **no seed-noise null** |
| Meta-learned multi-teacher weights | MMKD ([arXiv 2306.06634](https://arxiv.org/abs/2306.06634)); MTKD-RL ([AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32990/35145)) | Validation-set or RL weighting of teachers | Classification-centred; no per-object significance test |
| In-sample per-instance teacher weights | M-AID / CA-MKD ([arXiv 2201.00007](https://arxiv.org/abs/2201.00007)) | Weights from each teacher's own ground-truth loss | In-sample, no noise model, same-domain teachers |
| Loss-supervised per-location view routing | ERA-Det, October 2026 ([arXiv 2610.26090](https://arxiv.org/abs/2610.26090)) | Per-location fusion weight supervised by the losses of two training-only single-view heads; lowers frame weight at night | Inference-time fusion, in-sample targets; we amortise out-of-fold, null-calibrated targets to gate KD only, and remove the head at test time |
| Label-free, calibrated teacher reliability | TrustMOPD ([arXiv 2609.23697](https://arxiv.org/abs/2609.23697)) | Token-level reliability calibrated across teachers (LLMs) | Restrict our claim to "for vision detection / enhanced views" |
| Per-image best view; training-only enhancement | AMIEOD ([arXiv 2605.06084](https://arxiv.org/abs/2605.06084)); SGLDet ([ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/2b00b3331bd0f5fbfdd966ac06338f6d-Paper-Conference.pdf)) | Image-level argmin-loss view selection; dense, ungated training-only guidance | Per-object, out-of-fold, significance-tested |
| Value of privileged information | PAC-Bayes LUPI value paper ([arXiv 2609.12891](https://arxiv.org/abs/2609.12891)) | Training-time metric of the potential gain from privileged information | Our "deterministic f(x) adds no information" framing is a special case they could also make: cite it |
| Raw-vs-raw seed null used to significance-test per-object teacher utility | **None found** | — | **The load-bearing novelty** |

The defensible delta claim reads: *prior work gates multi-teacher distillation with in-sample losses (M-AID, CA-MKD, PATT) or validation meta-gradients (MMKD, DIW), and uses cross-fitting only to de-bias targets (Dao et al.). We treat each enhancer as a treatment on a deterministic view of the same input, estimate its per-object effect out-of-fold, and significance-test it against a raw-vs-raw seed null before distilling* ([Paper 2 re-check](/home/user/lowlight/research_notes/Low%20light%20detection%20paper%20plans/paper2_novelty_recheck.md)). Only the last clause, plus the low-light instantiation, is unclaimed. The expected killer review is no longer "M-AID with a better gate" but "M-AID plus Dao cross-fitting plus a variance normaliser". The rebuttal must be empirical: our gate has to beat the in-sample gate, an out-of-fold gate without the null, a DIW-style meta-gradient gate, and a raw-seed ensemble pushed through the identical gate. Novelty rises to about 5.5–6 only if C3 shows semi-supervised gains on two datasets and the seed-null ablation clearly wins.

The Related Work section must cite Dao et al., DIW, M-AID/CA-MKD, MMKD, ERA-Det, TrustMOPD, PATT ([arXiv 2609.02471](https://arxiv.org/abs/2609.02471)), Lopez-Paz et al.'s generalized distillation ([arXiv 1511.03643](https://arxiv.org/abs/1511.03643)), the PAC-Bayes LUPI paper, AMIEOD, SGLDet and TaskGuard ([arXiv 2609.08011](https://arxiv.org/abs/2609.08011)). No September–November 2026 arXiv paper distils enhanced-view teachers into a raw-input low-light detector. The nearest are uniform clean-teacher distillation at PRCV 2025 ([Springer](https://link.springer.com/chapter/10.1007/978-981-95-5755-4_9)) and C2FXNet's scene-expert mixture at ACM MM 2026 ([arXiv 2609.25693](https://arxiv.org/abs/2609.25693)).

## Published numbers predict small, specific wins rather than leaderboard jumps

### Paper 1 should yield a publishable answer under most outcomes

The zero-shot evidence is consistent and dataset-dependent. On **DarkFace, all 14 LoViF enhancers beat raw** with a frozen detector (0.207 → 0.226–0.262 mAP50). On **ExDark, only 3 of 14 beat raw, all by ≤ +0.009**, and artefact-heavy enhancers lose up to 0.14 ([CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/papers/Cheng_Generative_Low-Light_Enhancement_for_Computer_Vision_Revisiting_the_Gap_Between_CVPRW_2026_paper.pdf)). LIME-Eval's YOLOX-m on ExDark gives raw 49.2 against 45.9–47.6 for enhance-first P3 and 48.5–49.2 for augment-first P3 ([arXiv 2410.08810](https://arxiv.org/abs/2410.08810)). So much of the familiar "retraining on enhanced data loses" result may be an augmentation-order artefact. The underwater study found raw best for all seven detectors, with the gap widening for stronger backbones and longer schedules ([arXiv 2311.18814](https://arxiv.org/abs/2311.18814)). That predicts Paper 1's scaling result: enhancement utility should fall as data and capacity grow.

| Dataset | P0 Δ vs raw | P2 Δ vs P1 | P3 Δ vs P1 | Residual (P1 − P3) |
|---|---|---|---|---|
| DarkFace | Positive, about +2 to +5 AP50 | Large negative (inferred) | Unknown; plausibly ≈ 0 to slightly positive (no published number) | — |
| ExDark | ≈ 0 ± 1, enhancer-dependent; −10 to −14 for artefact-heavy enhancers | Large negative, likely −5 to −20 mAP (inferred; never published) | 0 to −3 enhance-first; ≈ 0 to −0.7 augment-first | Small |
| All | — | Adaptable penalty (P3 − P2) dominates | — | Main risk: ≈ 0 under proper controls |

Our own earlier pilot points the same way, though it used a custom 70/20/10 split, 737 test images, one seed and the buggy SCI output. On YOLOv8n, the adaptable share of the P2 penalty was 58–90% across enhancers. Small objects gained relative AP under Zero-DCE++ (+18%) and KinD++ (+15%) while medium and large objects lost, which is the opposite of LR-YOLO's synthetic trend ([pilot reanalysis](/home/user/lowlight/research/analysis/reanalysis_output.txt); [LR-YOLO](https://arxiv.org/html/2510.24232v3)). These are hypotheses to re-test on the standard splits, not results.

Five outcomes are each publishable if the CIs are tight. The strongest is a protocol × darkness interaction that survives within-dataset darkness strata. A residual penalty ≈ 0 under augment-first and quantisation controls would refute the "information loss" narrative. A monotone decline of utility with data fraction and with unfreezing would be practically useful, ideally as a crossover rule of the form "enhance only below N images or with a frozen backbone". Interventions in which luminance-only reproduces most of the P0 gain and the noise band explains most of the harm, with CKA predicting per-object ΔAP, would supply the mechanism. A real-data size effect in either direction is a bonus. Four outcomes weaken the paper: all P1/P3/P4 differences overlapping while the scaling curve stays flat; the interaction disappearing under within-dataset stratification; CKA failing to predict ΔAP; and Kumari et al. or an unindexed TCSVT paper turning out to run a multi-protocol grid.

**The minimum Q1 result set** has three parts. First, the full P0–P4 matrix on three datasets plus within-dataset strata, three detector families, 5–6 enhancers spanning artefact levels plus gamma and identity controls, three seeds, paired-bootstrap CIs and a mixed-effects test of the interaction. Second, the P2/P3 decomposition under both augmentation orders. Third, at least one of the scaling or intervention pillars done fully; doing both makes acceptance comfortable ([Paper 1 re-check](/home/user/lowlight/research_notes/Low%20light%20detection%20paper%20plans/paper1_novelty_recheck.md)).

### Paper 2 will beat raw training; whether the gate beats ungated distillation is a coin flip

| Comparison (YOLOv8n/s, raw input) | Expected gain, mAP50 | Main driver |
|---|---|---|
| vs plain raw training, ExDark-MAET | +2 to +4 | YOLOv8m teachers plus multi-view training |
| vs mixed raw+enhanced (P4) | +1 to +2 | Distillation |
| vs AMIEOD recipe ported to YOLOv8 | +0.5 to +1.5; may be ≈ 0 | — |
| vs SGLDet reimplementation | +0.5 to +2 | — |
| Gate vs ungated multi-teacher KD | **+0.3 to +1.0**; on ExDark ≈ +0.5 with CIs straddling zero | The gate itself |
| vs strongest zero-cost control | +0.8 to +1.5 on ExDark; **+1.5 to +3 on DarkFace** | Small, dark faces |

These ranges are extrapolations from published deltas, not measurements ([Paper 2 re-check](/home/user/lowlight/research_notes/Low%20light%20detection%20paper%20plans/paper2_novelty_recheck.md); [literature audit 05](/home/user/lowlight/research/literature/05_paper2_distillation_novelty_and_design.md)). The strongest cheap competitor is multi-view training. AMIEOD's model gets **81.9 mAP50 on raw ExDark input versus 76.5 plain** (YOLOv3, MAET split). Mixed training without its regression loss gets only 76.9–78.3, and AMIEOD's gain shrinks on stronger YOLOs: +2.7 (v5s), +0.5 (v7), +1.5 (v9-s) ([arXiv 2605.06084](https://arxiv.org/abs/2605.06084)). Gating is plausible because ungated illumination consistency can hurt (YOLA-Naive loses 0.9 on ExDark/TOOD) ([arXiv 2410.18398](https://arxiv.org/abs/2410.18398)), and per-instance weighting has added +2.5 in 3D detection ([arXiv 2402.18493](https://arxiv.org/abs/2402.18493)). Raw and enhanced views of the same image, though, differ far less than events and frames, where ERA-Det's routing halved the accuracy loss. Design choices add mAP in this order: teacher capacity (YOLOv8m/l, with an equally large raw teacher as control), then multi-view training as the base recipe, then masked FPN feature distillation plus cls/DFL logit distillation, then photometric augmentation, then two or three diverse enhancers, and the gate last.

DarkFace is where to look for a headline. SGLDet reports 59.7 → 76.6 there with DSFD trained from scratch, against 76.4 → 78.6 on ExDark ([ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/2b00b3331bd0f5fbfdd966ac06338f6d-Paper-Conference.pdf)). A paper whose headline is "a measured, null-calibrated map of where enhancement helps, plus a zero-cost detector that beats AMIEOD-raw and SGLDet on DarkFace by ≥ 2" is viable at ACM MM. A paper whose headline is the gate's ExDark delta is not viable at ICCV.

Published YOLOv8 ExDark numbers range from 71 to 80 mAP50 under incompatible protocols: DetectionBench YOLOv8n 71.29, YOLOv8x 75.4 ([HF model card](https://huggingface.co/dronefreak/exdark-yolov8n)), and a repository claiming 80.17 for YOLOv8s ([GitHub](https://github.com/riyadijiskandar-beep/DAEG-YOLO)). So **every comparison row must be our own run on the MAET split with three seeds**. Published YOLOv3/TOOD numbers go in a separately labelled block. Week-one sanity targets: YOLOv8n raw on ExDark-MAET about 70–76, YOLOv8n raw on DarkFace-YOLA about 55–65, and the YOLOv8m raw teacher at least +3 over YOLOv8n. If the teacher margin is smaller, we switch to YOLOv8l, because otherwise there is no distillation headroom.

## Paper 1 plan: about 670 trainings across three datasets, three detector families and five protocols

Working title: *When Does Low-Light Enhancement Help Detection? A Protocol × Darkness Study with Controlled Interventions.* The protocols are:

| Protocol | Train on | Test on |
|---|---|---|
| P0 | Normal-light pretrained detector (no training) | Raw vs enhanced, zero-shot |
| P1 | Raw | Raw |
| P2 | Raw | Enhanced |
| P3 | Enhanced | Enhanced |
| P4 | Raw + enhanced mix | Raw and enhanced |

The adaptable penalty is P3 − P2 and the residual penalty is P1 − P3. All GPU-hour figures are estimates for one 24 GB card and must be replaced after timing one epoch per detector in week one ([Paper 1 execution plan](/home/user/lowlight/research_notes/Low%20light%20detection%20paper%20plans/paper1_execution_plan.md)).

### Datasets, splits and preparation

| Dataset | Split used | Size | Classes | Role / settings |
|---|---|---|---|---|
| **ExDark-MAET** (primary) | MAET 80/20: `main/train.txt` / `main/val.txt` used as **test** | 5,896 / 1,467 | 12 | Headline benchmark; imgsz 640. Carve a class-stratified 10% internal val (≈590, seed 0) for sanity only; report the **last epoch** for every arm ([ICCV_MAET](https://github.com/cuiziteng/ICCV_MAET)) |
| ExDark-official (secondary, B1) | Column 5 of `imageclasslist.txt` | 3,000 / 1,800 / 2,563 | 12 | Comparability with YOLA/FRBNet/Wu et al.; lighting analysis second source ([ExDark repo](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)) |
| **DarkFace** | YOLA `train.txt`/`val.txt` | 5,400 / 600 (+9,000 or 10,400 unlabelled for Paper 2; count on download) | face | Brightest P0 gains; Ultralytics 1024 long side, MMDetection 1333×800 ([YOLA repo](https://github.com/MingboHong/YOLA)) |
| **NOD** | Official COCO JSONs, combined Sony+Nikon | 5,777 / 721 / 721 images; 37,456 / 4,616 / 4,679 boxes | person, bicycle, car | Extreme dark; Sony test carries per-box `isextreme` (815 extreme vs 1,012 non-extreme); pre-resize to 1,333 px long side before enhancing ([NOD annotations](https://github.com/igor-morawski/NOD/blob/main/annotations/README.md)) |
| BDD100K-night (B2) | `timeofday == "night"`; ≈10k stratified train subset; all val-night for test | counts to compute | 10 (9 for P0) | Driving generality; spot-check 100 images for label noise ([issue #20](https://github.com/bdd100k/bdd100k/issues/20)) |
| LLVIP-visible (B3) | `Train_LLVIP_vi` / `Test_LLVIP_vi`, post-2023 labels | 15,488 pairs | person | Labels drawn on infrared, so it tests "invisible objects" ([LLVIP](https://github.com/bupt-ai-cz/LLVIP)) |
| **COCO darkness ladder (A12, new)** | COCO val2017 for P0/P2; a fixed 10k COCO-train subset for P1/P3 | 5,000 val; 10,000 train | 80 | Breaks the darkness/dataset confound: four EV-drop levels with Poisson–Gaussian RAW-style noise, in the style of [arXiv 2605.22455](https://arxiv.org/abs/2605.22455) |

**Preparation, done once.** Convert every set to COCO JSON (for pycocotools, MMDetection and TIDE) and YOLO txt (for Ultralytics), with one canonical image_id ↔ filename table, and build a `meta.csv` holding split, ExDark lighting code, indoor/outdoor flag, and Rec.601 mean and 5th-percentile luminance per image. Assert that MAET's 7,363 names are a subset of `imageclasslist.txt`. ExDark has mixed extensions (6,195 .jpg, 496 .png, 424 .JPEG and others), so normalise EXIF orientation once and re-save as lossless PNG; checking that raw-PNG P1 equals raw-original P1 is the identity-I/O control. ExDark filenames are class-sorted (the first 736 are Bicycle plus Boat), so never use Ultralytics `fraction` and build stratified subset lists ourselves. NOD image and annotation ids are re-enumerated per JSON, so join on filenames.

**P0 class mapping.** ExDark maps to COCO-80 as Bicycle 1, Boat 8, Bottle 39, Bus 5, Car 2, Cat 15, Chair 56, Cup 41, Dog 16, Motorbike 3, People 0 and Table 60, with a robustness row that excludes Table and Cup; NOD's person, bicycle and car map to 0, 1 and 2. Predictions of unmapped COCO classes are discarded before evaluation. DarkFace has no COCO class, so P0 there uses YOLOv8n and Faster R-CNN trained on WIDER FACE (one run each, mirroring the TPAMI survey's DSFD protocol).

**Licences.** ExDark's "Research Use Terms v1.0" (effective 1 October 2026) allow non-commercial research and the publication of aggregate results, but exclude the underlying photographs ([DATASET_TERMS](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Dataset/DATASET_TERMS.md)). We release splits and predictions, not images. The DarkFace, BDD100K and LLVIP licences are unverified and must be checked on download.

### Enhancers and controls

All enhancers take sRGB in [0,1] with no mean/std normalisation. The real hazards are which tensor gets saved, the padding geometry, and per-dataset weight choice. One wrapper, `enhance.py`, handles all methods: it reflect-pads to the method's multiple and never crops, runs in eval mode with a fixed seed, crops back, clamps, saves lossless PNG with the same basename, asserts that the output shape equals the input shape, and logs runtime, mean luminance, percentage of clipped pixels and the SHA-1 of each output.

| Arm | Weights | Output to save | Pitfall to avoid |
|---|---|---|---|
| LIME | none | enhanced image | pvnieo `demo.py` defaults to DUAL; **add `-ul`** (`-g 0.6 -l 0.15 -s 3`) ([demo.py](https://github.com/pvnieo/Low-light-Image-Enhancement/blob/master/demo.py)) |
| Zero-DCE++ (core) | `Epoch99.pth` | `enhanced_image` | Official script **crops** to multiples of 12; pad instead ([lowlight_test.py](https://github.com/Li-Chongyi/Zero-DCE_extension/blob/main/Zero-DCE%2B%2B/lowlight_test.py)) |
| SCI (core) | `medium.pt` (ablate `difficult.pt`) | **`r`, not `i`** | Our earlier results saved the illumination map `i`; check `r ≥ x` on 20 images ([SCI model.py](https://github.com/vis-opt-group/SCI/blob/main/CVPR/model.py)) |
| Retinexformer (core) | LOL-v1 (ablate LOL-v2-real) | model output | Pad to ×4; **never `--GT_mean`**; no self-ensemble ([repo](https://github.com/caiyuanhao1998/Retinexformer)) |
| PairLIE (core) | `weights/PairLIE.pth` | **`I = L^0.2·R`** from folder `I/` | `eval.py` writes L, R, I and D ([eval.py](https://github.com/zhenqifu/PairLIE/blob/main/eval.py)) |
| SNR-Aware | LOLv1 checkpoint | model output | Test scripts are paired; write an unpaired loop that rebuilds the SNR map |
| KinD++ | official TF checkpoint | final output | Separate TF ≥ 2 environment |
| HVI-CIDNet | `LOLv2_syn/generalization.pth` | eval output | `--unpaired --custome --alpha 1.0 --gamma 1.0` ([repo](https://github.com/Fediory/HVI-CIDNet)) |
| LightenDiffusion (9th, P0/P2) | released checkpoint | eval output | Stochastic: fix seed and sampler steps |
| C-id (control) | — | raw through the same I/O wrapper | — |
| C-q (control) | — | core-2 outputs stored as 8-bit vs 16-bit; plus a gain-requantise-divide variant on raw | — |
| C-γ (control) | — | per-image global gamma, matched by bisection to the enhancer's mean luminance | One per enhancer at P0/P2; core-2 only at P3; doubles as the gain-only intervention |

The enhancer groups are:

| Group | Members |
|---|---|
| Core-4 | SCI (fixed), Zero-DCE++, Retinexformer, PairLIE (all torch, so all usable augment-first) |
| Core-2 | SCI and Retinexformer, for the most expensive experiments |
| Full-8 | Core-4 + LIME, SNR-Aware, KinD++, HVI-CIDNet |

The MAET archive's pre-enhanced `IMGS_ZeroDCE`/`IMGS_Kind` folders serve only as cross-checks. Their settings are unknown.

### Detectors and frozen recipes

| Detector | Init | imgsz | Schedule | Batch / optimiser | Notes |
|---|---|---|---|---|---|
| YOLOv8n (s, m for capacity) | `yolov8{n,s,m}.pt` | 640; DarkFace 1024 | 100 epochs, `close_mosaic=10`, `patience=0` | 16 (m: 8, `nbs=64`); **`optimizer=SGD, lr0=0.01, momentum=0.937`** | Save last epoch; default HSV/mosaic/flip |
| RT-DETR-l | `rtdetr-l.pt` (53.0 COCO AP) | 640; DarkFace 1024 (960 if OOM) | 72 epochs | 8; AdamW 1e-4 | AMP on |
| Faster R-CNN R50-FPN | COCO 1x checkpoint (37.4 AP) | 1333×800 | 12 epochs, steps 8/11 | 4; SGD 0.005 | Geometric-only base pipeline |
| DINO-4scale R50 (B5) | COCO 12e checkpoint (49.0 AP) | default | 12 epochs | 2; AdamW 1e-4 × 2/16 | Replaces the 50-epoch Deformable DETR |
| YOLO11n/s (B4) | `yolo11{n,s}.pt` | as YOLOv8 | as YOLOv8 | as YOLOv8 | Replication |

Two Ultralytics defaults would silently corrupt the study. First, `optimizer=auto` switches between AdamW and MuSGD depending on iteration count, which changes the optimiser across data fractions. Second, `fraction` takes the first k files ([trainer.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py); [base.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/base.py)). Set the optimiser explicitly and use our own lists. Seeds are {0, 1, 2} with deterministic mode in both frameworks. Log the torch, CUDA and cuDNN versions and the git hash.

**Augmentation orders.** Enhance-first points the dataset at pre-enhanced PNGs. Augment-first subclasses Ultralytics' `DetectionTrainer.preprocess_batch` and `DetectionValidator.preprocess` (or MMDetection's `DetDataPreprocessor`) to apply the frozen enhancer to the augmented [0,1] batch on the GPU. That is feasible only for the torch core-4.

**Backbone freezing.** Use `freeze=10` for YOLOv8 and `frozen_stages=4` with `norm_eval=True` for Faster R-CNN. These recipes are design proposals; freeze them after a one-seed pilot on ExDark-MAET raw.

### Every run, with costs

| ID | Experiment | Factors | Runs | GPU-h (est.) |
|---|---|---|---|---|
| A0 | Enhancement generation | 9 methods × ExDark-MAET, DarkFace, NOD (≈20.6k images) | — | ≈40 (LIME on CPU) |
| A1 | **P1** raw → raw | 3 datasets × {YOLOv8n, FRCNN, RT-DETR-l} × 3 seeds | 27 | ≈90 |
| A2 | **P0** zero-shot | COCO checkpoints on ExDark and NOD × {raw, 9 enhancers, C-id, 9 C-γ}; DarkFace via 2 WIDER-FACE detectors | 2 + ≈330 evals | ≈25 |
| A3 | **P2** raw-trained, enhanced test | 27 A1 models × {9 enhancers, C-id, C-q, 9 C-γ} | ≈540 evals | ≈15 |
| A4 | **P3 enhance-first** | YOLOv8n: 12 arms (full-8, C-id, C-q, C-γ core-2) × 3 datasets × 3 seeds = 108; FRCNN and RT-DETR-l: (core-4 + C-id) × 3 × 3 = 45 each | 198 | ≈500 |
| A5 | **P3 augment-first** | YOLOv8n core-4 × 3 datasets × 3 seeds = 36; FRCNN core-2 × {ExDark, DarkFace} × 3 = 12 | 48 | ≈100 |
| A6 | Photometric-off | YOLOv8n {raw, core-4}, HSV = 0 × {ExDark, DarkFace} × 3 seeds | 30 | ≈45 |
| A7 | **P4 mixed** (p = 0.5 per image, resampled each epoch) | YOLOv8n core-4 × 3 × 3 = 36; FRCNN and RT-DETR-l core-2 × 3 × 3 = 18 each; tested on raw and enhanced | 72 | ≈190 |
| A8 | **Data scaling** | YOLOv8n × {ExDark, DarkFace} × {raw P1, SCI P3, Retinexformer P3, SCI P4} × {10, 25, 50%} × {fine-tuned, frozen} × 3 seeds (5 at 10%), + frozen at 100% | ≈190 | ≈90 |
| A9 | **Controlled interventions** | P0/P2: 4 interventions × full-8 × 3 datasets (inference); P3: YOLOv8n × core-2 × 4 interventions × {ExDark, DarkFace} × 3 seeds | 48 + ≈300 evals | ≈80 |
| A10 | **Capacity sweep** (augment-first, so it shares runs with Paper 2) | {YOLOv8s, YOLOv8m} × {raw, SCI, Retinexformer} × {ExDark, DarkFace} × 3 seeds, + P2 inference | 36 | ≈85 |
| A11 | Analyses: CKA/MMD features, per-object scores, TIDE, lighting and size | — | — | ≈15 |
| A12 | **Darkness ladder** (new) | P0/P2 on darkened COCO val at 4 levels × {raw, core-4} × 3 COCO detectors; P1/P3 YOLOv8n on a 10k train subset at the 2 extreme levels × {raw, SCI, Retinexformer} × 3 seeds | 18 + ≈60 evals | ≈40 |
| | **Total must-have** | | **≈670 trainings** | **≈1,220** |

The lean path saves about 480 GPU-hours and brings the core to about 740 h. It gives RT-DETR-l three seeds in A4 only on core-2 + C-id (one seed for the other two enhancers), runs A7's RT-DETR-l with one seed, and in A10 keeps YOLOv8s at three seeds and YOLOv8m at one. **Never cut seeds from YOLOv8n or from any ExDark/DarkFace cell in the headline heatmap.**

The nice-to-have tier, in priority order:

| ID | Experiment | GPU-h (est.) |
|---|---|---|
| B1 | ExDark official split, YOLOv8n P1 + P3 core-4, 3 seeds + P2 | ≈15 |
| B2 | BDD100K-night, YOLOv8n P0–P3 core-4, 3 seeds | ≈60 |
| B3 | LLVIP-visible, YOLOv8n P0–P3 core-4 | ≈40 |
| B4 | YOLO11n replication on ExDark + DarkFace | ≈40 |
| B5 | DINO P1 + P3 core-2, 1 seed | ≈60 |
| B6 | LightenDiffusion P3 | ≈15 + generation |
| B7 | Per-image oracle and a brightness-rule gate with regret rate | CPU |
| B8 | Faster R-CNN data scaling, ExDark, 1 seed | ≈40 |
| B9 | Enhancer weight-choice ablation (LOL-v1 vs v2-real) at P0/P2 | ≈3 |
| B10 | IQA (NIQE/BRISQUE/MUSIQ) vs ΔmAP over ≥50 cells, as hygiene only | CPU |

B7 matters most for Pattern Recognition, which tends to want a method angle.

### Mechanism analyses, specified exactly

Work per image x with enhancer output y = E(x) in float RGB, luminance Y = 0.299R + 0.587G + 0.114B, and ε = 1e-4. The four interventions are built as follows:

| Intervention | Construction | What it keeps from x |
|---|---|---|
| **Gain-only** | x_tone = clip(x · T(Y_x) / max(Y_x, ε)), where T histogram-matches Y_x to Y_y | Raw chromaticity, noise and detail |
| **Colour-only** | x_col = clip(Y(x_tone) · y / max(Y_y, ε)) | Luminance of x_tone; takes y's chromaticity |
| **Detail-only** | x_detail = x_tone − d(x_tone) + d(y) | Everything except the detail band |
| **Noise-only** | x_noise = x_tone − n(x_tone) + n(y) | Everything except the noise band |

The two bands come from a self-guided filter G. The noise band is n(z) = z − G_{r=1, e=1e-4}(z). The detail band is d(z) = z − G_{r=8, e=1e-2}(z) − n(z). Swapping all components back in must reconstruct y at ≥ 35 dB PSNR on ≥ 95% of images; report the remainder as "interaction". Leave-one-out variants, plus a 16-combination Shapley attribution at P0/P2, avoid assuming the effects are additive.

**CKA and MMD.** Hook YOLOv8n's backbone indices 2, 4, 6 and 9 and neck outputs 15, 18 and 21; Faster R-CNN's `layer1`–`layer4` and FPN P2–P6; and RT-DETR-l's HGNetv2 stages and hybrid-encoder outputs. Pool each image to 4×4, and RoIAlign 7×7 per ground-truth object. Compute minibatch linear CKA ([ckatorch](https://pypi.org/project/ckatorch/)), unbiased RBF-MMD² with a 1,000-permutation test, and per-object cosine distance. Link them to accuracy in two ways: Spearman ρ between layer-wise CKA drop and ΔmAP over about 120 cells, and a mixed model of per-object Δscore on feature distance with random intercepts for image and enhancer.

**Per-object regret.** Let s(o) be the highest same-class confidence matched to object o at IoU 0.5 (0 if missed). The object regret rate is the share of objects with s_E − s_raw < −0.1, with a threshold sweep. The image regret rate is the share of images whose per-image AP50 drops. Stratify both by COCO size, size quantile, lighting group and darkness quartile.

**TIDE and lighting groups.** TIDE covers P1/P2/P3 for core-4 × 3 detectors × 3 datasets ([tidecv](https://github.com/dbolya/tide)). ExDark lighting analysis merges four groups for inference (Low+Weak+Shadow; Ambient+Twilight; Object+Single+Screen+Window; Strong) and shows all ten conditions descriptively only, since the full dataset has just 56 Shadow and 165 Screen images.

### Statistics

Every comparison is paired at the image level. Run `COCOeval.evaluate()` once per run and test set. Then rerun a weighted copy of `accumulate` on resampled image multiplicities; with unit weights it must reproduce pycocotools to 1e-6. This makes B = 10,000 (headline cells) and 2,000 (grid cells) cheap on CPU.

**Seeds** are averaged inside each resample, giving the primary estimand. A two-level bootstrap goes in the supplement, along with each cell's seed standard deviation.

**Multiple comparisons.** Apply Holm within each family (for example, all enhancer-vs-raw tests for one dataset × detector × protocol). Report Benjamini–Hochberg q-values in the supplement, and dot Holm-significant cells in the heatmaps.

**Rankings** use `autorank` Friedman + Nemenyi critical-difference diagrams over 36 dataset × detector × protocol blocks ([autorank](https://github.com/sherbold/autorank)).

**The headline interaction** is a `statsmodels` mixed model, ΔmAP ~ protocol × darkness + (1|enhancer) + (1|detector). It is confirmed three ways: a permutation test over datasets (only 3–5 levels, which we state), the within-dataset quartile analysis, and the A12 ladder.

**Other rules.** Decomposition CIs bootstrap P3 − P2 and P1 − P3 on identical indices. Correlations report n and bootstrap CIs. We pre-register |Δ| < 0.5 mAP50 as "no practical effect".

### Tables, figures, ablations and outline

**Ablations listed in the paper** are augmentation order, photometric augmentation on/off, enhancer weights (LOL-v1 vs v2-real), SCI medium vs difficult, checkpoint rule (last vs best on held-out val), DarkFace image size (640 vs 1024), 8- vs 16-bit storage, P0 mapping with and without Table/Cup, and evaluation toolkit (pycocotools vs Ultralytics mAP).

| Item | Content |
|---|---|
| **Fig. 1** teaser | ExDark Ambient and DarkFace examples with raw/SCI/Retinexformer detections; 2×2 protocol × darkness ΔmAP mini-plot with CIs |
| **Fig. 2** protocol diagram | P0–P4 pipelines, both augmentation orders, where controls enter, and the decomposition arrows (P3 − P2 adaptable, P1 − P3 residual) |
| **Fig. 3** core heatmaps | Rows: enhancers + controls. Columns: detector × dataset (ordered by darkness). One panel each for P0, P2, P3, P4; diverging map centred at 0; Holm dots |
| **Fig. 4** decomposition | Stacked adaptable/residual bars per dataset × detector for core-4, enhance-first vs augment-first, with CIs |
| **Fig. 5** scaling | ΔmAP(E − raw) vs data fraction (log-x), fine-tuned solid vs frozen dashed, ExDark and DarkFace, 95% bands |
| **Fig. 6** interventions | ΔmAP for tone/colour/detail/noise/full per enhancer at P0, P2 (P3 for core-2), with leave-one-out or Shapley bars |
| **Fig. 7** representation | (a) Layer-wise CKA curves; (b) CKA drop vs ΔmAP scatter over ≥ 50 cells with ρ and CI; (c) per-object distance vs Δscore |
| **Fig. 8** where it helps | (a) Size; (b) ExDark lighting groups; (c) NOD extreme vs non-extreme; (d) darkness-quartile × protocol; (e) A12 darkness ladder |
| **Fig. 9** CD diagrams | Friedman/Nemenyi per protocol (at least P0 and P3) |
| **Fig. 10** qualitative | 6 images × raw, 8 enhancers and 4 interventions, including noise-induced false positives and colour-shift misclassifications |
| **Tab. 1** datasets | Splits, images/boxes, classes, mean/p5 luminance, licence, COCO mapping |
| **Tab. 2** enhancers | Venue, type, weights, I/O settings, runtime per megapixel, mean ΔY, plus the controls |
| **Tab. 3** main results | mAP50 and mAP50-95 for P1 and best/median/worst enhancer under P0/P2/P3/P4 per dataset × detector, with CIs and Holm marks |
| **Tab. 4** comparability | Our raw numbers next to published MAET/YOLA/FRBNet/SGLDet/AMIEOD baselines, never mixing protocols |
| **Tab. 5** order and controls | Raw, C-id, C-q, C-γ, enhance-first, augment-first, photometric-off for core-4 (YOLOv8n, FRCNN) |
| **Tab. 6** capacity and regret | ΔmAP and object/image regret by YOLOv8 n/s/m |
| Supplement | Full grids, per-class, all ten conditions, TIDE, all seeds, IQA correlations, weight ablation, oracle/gated selection, Ultralytics-mAP mirror |

The IEEE double-column page budget runs to about 12 pages plus references:

| Section | Pages |
|---|---|
| Introduction | 1.25 |
| Related work | 1.25 |
| Study design | 2.5 |
| Protocol × darkness results | 1.75 |
| Decomposition | 1.0 |
| Scaling and capacity | 1.0 |
| Mechanisms | 1.5 |
| Where it helps | 0.75 |
| Discussion with a practitioner "when to enhance" table | 0.75 |
| Conclusion | 0.25 |
| References (60–80) | ≈1.5–2 |

**IEEE TMM is the first choice.** It published both directly adjacent LLIE-for-detection papers, Morawski et al. ([arXiv 2412.07693](https://arxiv.org/abs/2412.07693)) and AMIEOD, so its reviewers will value P2, scaling and intervention evidence. **TCSVT is equally plausible.** The 2025 IEEE APC list suggests 11 free pages there (unconfirmed), and open access costs $2,800 in 2026 ([IEEE APC list](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/2026-07-30-IEEE-Article-Processing-Charges-List.pdf)). Pattern Recognition is the alternative if B7 supplies a method angle. TPAMI is a stretch: Pei et al.'s analysis-only TPAMI paper spanned several degradations ([arXiv 1810.05552](https://arxiv.org/abs/1810.05552)), whereas ours covers low light only. Current TMM, TCSVT and Pattern Recognition page limits could not be retrieved, so confirm them on the IEEE Author Center and in the PR Guide for Authors before formatting.

The reproducibility package contains the split files, `enhance.py` with commands and output hashes, all prediction JSONs, the bootstrap/Holm/autorank notebook, the frozen configs and `runs.csv`.

## Paper 2 plan: about 650 GPU-hours behind three go/no-go gates

Working title: *Distilling Computable Privilege: Seed-Calibrated, Out-of-Fold Enhancement Utility for Raw-Input Low-Light Detection.* Shorter alternative: *When Does Enhancement Help? Null-Calibrated Per-Object Distillation for Low-Light Detection.* The abstract's final sentence, "the gain shrinks with data size and concentrates on dark, small objects", stays only if that prediction holds ([Paper 2 re-check](/home/user/lowlight/research_notes/Low%20light%20detection%20paper%20plans/paper2_novelty_recheck.md)).

### Method specification on Ultralytics' native distillation

Ultralytics now ships detection distillation. A `DistillationModel` hooks the three neck layers feeding `Detect` (indices 15, 18, 21, 22 for YOLOv8). It uses a 1×1–ReLU–1×1 projector and a teacher-score-weighted L2 loss scaled by `dis = 6.0`. The teacher runs under `no_grad` on the student's batch, and `fuse()` strips the wrapper ([distill_model.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py); [KD guide](https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/knowledge-distillation.md)). The guide does not support RT-DETR or cross-family distillation. We pin `ultralytics==8.4.201` (commit `73c60b7`) and subclass the wrapper as `GatedDistillationModel` with a `GatedDistillTrainer`, without editing site-packages. A unit test checks that the exported YOLOv8n has exactly 3,157,200 parameters. The code is AGPL-3.0, so our release must be too ([Paper 2 execution plan](/home/user/lowlight/research_notes/Low%20light%20detection%20paper%20plans/paper2_execution_plan.md)).

| Component | Exact specification | Default / sweep |
|---|---|---|
| Teacher inputs | Enhancer applied **online on the GPU** to the already-augmented raw batch, x_k = E_k(batch), under `no_grad` (augment-first; keeps mosaic and affine geometry aligned) | Ablate offline enhance-first once |
| Per-object loss ℓ_t(o) (C1) | (a) Assignment-based: TAL `get_assigned_targets_and_loss`, sum 7.5·box + 0.5·cls + 1.5·dfl over anchors with `target_gt_idx == o`; or (b) ℓ = −log(max_p[score·IoU] + ε) over same-class predictions | Choose in week 2 by test–retest Spearman between raw seeds |
| Null scale (C1) | 9 bins (size tercile × local-luminance tercile, luminance over box ×1.5); s_bin = 1.4826·MAD(ℓ_raw1 − ℓ_raw2) | — |
| Utility z (C1) | z_k(o) = (ℓ̄_raw(o) − ℓ_k(o)) / (s_bin·√(3/4)); the √(3/4) makes var = 1 under H0 | Calibration: raw seed 3 as a fake enhancer must give P(z > 2) ≈ 2.3% |
| Gate (C1) | g_k = σ((z_k − τ)/T_g); softmax(z_k) renormalisation among passing k; w_raw = max(0, 1 − Σg_k) | τ = 2, T_g = 0.5; sweep τ ∈ {0, 1, 1.5, 2, 3}, T_g ∈ {0.25, 0.5, 1, hard}, plus BH-FDR q = 0.1 hard gate |
| Feature KD (C2) | PKD per-channel standardisation; box mask per object on P3–P5, FGD scale weight 1/(h·w); Σ_t w_t(o)·‖M ⊙ (N(F̂_S) − N(F_t))‖²/C; background term from raw teacher only | λ_fg = 6 (sweep 1/3/6/10), λ_bg = λ_fg/2 ([PKD config](https://github.com/open-mmlab/mmrazor/blob/main/configs/distill/mmdet/pkd/pkd_fpn_faster-rcnn_r101_faster-rcnn_r50_2x_coco.py); [FGD](https://github.com/yzd-v/FGD/tree/master/configs/distillers/fgd)) |
| Logit KD (C2) | Student TAL positives mapped to objects; gated sigmoid-BCE cls KD (T = 1); gated DFL KL with τ_LD; raw-only cls KD on background at 0.25 | λ_cls = λ_dfl = 1 (sweep 0.5/1/2); τ_LD ∈ {1, 5, 10} |
| Total loss | L_det + λ_fg·L_feat + λ_bg·L_bg + λ_cls·L_cls + λ_dfl·L_dfl [+ λ_u·L_util] | KD weights warmed up over 3 epochs; SGD lr0 0.01, 100 epochs, 640 |
| Utility head (C3) | RoIAlign 7×7 on student P3–P5 → 2-layer MLP (256) → ẑ ∈ ℝ^K; Huber (δ = 1) to z; stop-grad into detector; deleted at export | λ_u = 1 |
| Unlabelled KD (C3) | Pseudo-boxes from full-data raw YOLOv8m teachers fused by WBF, conf ≥ 0.5; gate σ((ẑ − τ)/T_g); feature + cls KD only, no box loss; 1:1 labelled:unlabelled batches | Sweep conf 0.3/0.5/0.7 |
| Compute saving | Skip an enhanced teacher's forward pass when no object in the batch has g_k > 0.05 | Expected −30–50% KD cost (estimate) |

### Datasets, teachers and folds

| Dataset | Split | imgsz | Role |
|---|---|---|---|
| ExDark (12 classes) | MAET 80/20 (5,896/1,467); YOLA/official split as a supplement row | 640 | Standard benchmark; all controls and ablations |
| DarkFace | YOLA 5,400/600 + unlabelled pool; never the 4,000-image challenge hold-out (labels not public) | 1024 (ablate 640) | Headline dataset; C3 |
| NOD | Official COCO JSONs (5,777/721/721; verified for Paper 1) + 0.9k unannotated Sony images | 640, after pre-resizing to 1,333 px | Third dataset, extreme dark |
| BDD100K-night | ≈10k night-train subset; val-night test; night-train as unlabelled pool | 640 | Cross-domain unlabelled pool for C3; optional fourth labelled set |

Folds are drawn from the training split only, stratified by image darkness tercile and class presence, and saved as `folds.json`. Hyperparameters are tuned on fold-0 validation within the training set, never on test.

| Teacher | Model / input | Purpose | Runs per dataset |
|---|---|---|---|
| R1, R2 | YOLOv8m raw, seeds 0/1 | Raw baseline for z; raw KD teacher | 3 folds × 2 + 2 full |
| R3 | YOLOv8m raw, seed 2 | Null calibration; raw-seed-ensemble control | 3 folds + 1 full |
| E1 | YOLOv8m on Zero-DCE++ (curve-based) | Enhanced teacher | 3 folds + 1 full |
| E2 | YOLOv8m on Retinexformer (LOL-v1; pick by fold-0 P3 mAP) | Enhanced teacher | 3 folds + 1 full |
| E3 | YOLOv8m on corrected SCI `r` | Enhanced teacher | 3 folds + 1 full |
| E1′ | YOLOv8m Zero-DCE++, seed 1 | Test–retest reliability of z | 3 folds (drop on DarkFace if short of time) |

K = 3 is the default. K = 5 is an ExDark ablation with E1 only, and K = 1 (in-sample) is the M-AID-style baseline. Fold runs train on two-thirds of the data. ExDark costs about 20 run-equivalents (≈50–70 GPU-h), DarkFace at 1024 px about 90–120 GPU-h, and NOD about 50. Train teachers augment-first with the same 100-epoch recipe as Paper 1, so that their input distribution matches the online-enhanced inputs they see during distillation.

### Every baseline

| # | Baseline | Status | Reproduction |
|---|---|---|---|
| B1 | Raw YOLOv8n | Must | Shared with Paper 1 A1 |
| B2 | Raw + extra epochs matching KD updates (≈130) | Must | `epochs=` |
| B3 | Photometric augmentation (`hsv_v` 0.7, gamma ∈ [0.4, 2.5], Poisson–Gaussian noise) | Must | Custom transform |
| B4 | P4 mixed training over the 3 enhancers, p = 0.5 | Must | Online enhancer in the dataloader |
| B5 | AMIEOD-style multi-view + per-image best-view regression, raw inference, ported to YOLOv8n; plus official AMIEOD YOLOv3 on MAET to anchor 81.9 | Must | [AMIEOD code](https://github.com/scujayfantasy/AMIEOD) (released; `train_AMIEOD_stage1.py`, `train_stage2.py`) |
| B6 | SGLDet reimplementation (training-only decoder, λ = 0.01, SCI-phase + denoised-amplitude target), labelled "our reimpl." | Must | [No code released](https://github.com/gw-shin/SGLDet); sanity: YOLOv3 on MAET ≈ 78.6 |
| B7 | Single raw-teacher KD (R1) with our C2 losses | Must | Gate ≡ raw |
| B8 | Ungated multi-teacher KD (w = 1/4 each) | Must | Our code |
| B9 | Random gate with on-rate matched per size × luminance bin | Must | Shuffle z within bins |
| B10 | In-sample M-AID/CA-MKD gate (K = 1, softmax over −ℓ, no null) | Must | Our code |
| B11 | Raw-seed-ensemble gate (R1/R2/R3 through the identical gate) | Must | Our code |
| B12 | Native Ultralytics KD (`distill_model=R1.pt`, `dis = 6`) | Must | Stock |
| **B16** | **DIW-style meta-gradient per-object gate** (one-step detached update evaluated on a fold-internal meta set) | **Must (added by re-check)** | Our implementation of [DIW](https://arxiv.org/abs/2608.24904) |
| **B17** | **Out-of-fold gate without the null** (K = 3, raw Δℓ) | **Must (added by re-check)** | Our code |
| B13 | YOLA illumination module ported to YOLOv8n | Nice; must if targeting ICCV | [YOLA](https://github.com/MingboHong/YOLA) |
| B14 | Enhance-then-detect cascades (YOLOv8n P3 per enhancer) | Must | Shared with Paper 1 A4 |
| B15 | FeatEnHancer, PE-YOLO, FRBNet, IA-YOLO, DAI-Net, MAET, TorchAdapt | Cite at their own protocols in a separate block | — |

### Ablations, analyses and statistics

**Ablation table A (ExDark-MAET, YOLOv8n; rows marked \* run with three seeds):**

| Factor | Settings |
|---|---|
| Gate type\* | None (B8), random (B9), in-sample (B10), raw-seed ensemble (B11), DIW-style (B16), out-of-fold without null (B17), **ours**; oracle in the supplement |
| Cross-fitting\* | K = 1, 3, 5 |
| Null normalisation\* | None, global scale, per-bin MAD, per-bin MAD + √(3/4) |
| Gate shape | τ and T_g sweeps; FDR hard gate |
| Teachers | Raw only; each enhancer alone; +2; +3; corrected vs uncorrected SCI |
| Teacher size | v8s / v8m / v8l (v8l on raw + best enhancer only) |
| Student size | v8n / v8s |
| Teachers used as targets | Full-data vs out-of-fold |
| C2 loss form | MSE, PKD, CWD (τ = 1), native L2; feature-only / logit-only / both; ± DFL KD; ± background term |
| FPN levels | P3, P4, P5, P3+P4, all; size-assigned vs all-level masks |
| Loss weights | λ_fg, λ_cls and τ_LD sweeps |
| Base recipe | Plain vs AMIEOD-style multi-view |
| Augmentation order | Online augment-first vs offline enhance-first |

**Ablation table B (DarkFace, C3)** varies C3 (off, on with labelled data only, on with unlabelled data), the unlabelled amount (0/25/50/100%, plus BDD-night as a cross-domain pool), pseudo-box confidence (0.3/0.5/0.7) and stop-gradient (on/off).

The head itself is scored three ways: Spearman ρ(ẑ, z) on held-out-fold objects, AUROC for z > τ, and the recovery ratio (AP_label-free − AP_ungated)/(AP_GT-gate − AP_ungated).

**Analysis experiments.** Each tests one of the theory's predictions.

| Experiment | Prediction tested | Design |
|---|---|---|
| Data scaling | Gain shrinks with data (P1) | 10/25/50/100% of ExDark for raw, ungated KD and ours; 3 seeds at 25% and 100%, 1 seed otherwise |
| Gain by size and luminance | Gain concentrates on dark, small objects (P2) | COCO S/M/L and local-luminance terciles |
| Ensemble decomposition | Part of the gain is ensembling (P3) | Δ_total = (B11 − no-KD) [ensembling/capacity] + (ours − B11) [enhancement-specific] |
| z reliability | z is a stable signal | ρ(z_E1, z_E1′) and the R3 null false-positive rate |
| TIDE | — | Raw vs ours |

**Statistics.** Every main-table row runs seeds 0, 1 and 2 and reports mean ± std of mAP50 and mAP50-95. Paired image-level bootstrap uses B = 2,000 (10,000 for headline comparisons) with percentile 95% CIs of Δ, and Holm correction across the roughly 14 baseline comparisons per dataset.

**Efficiency table** columns: params, GFLOPs at 640 (YOLOv8n 8.9; s 28.8; m 79.3), FPS at batch 1 in FP16 (PyTorch and TensorRT), latency, extra training memory, and **training GPU-hours including teachers**. The rows compare raw, ours (identical inference), enhancer + detector cascades, YOLA-v8n, reported FeatEnHancer/PE-YOLO numbers, and AMIEOD with vs without its ResNet-50 selector.

### Figures, tables, outline and venue

| Item | Content |
|---|---|
| **Fig. 1** teaser | One ExDark image with a significantly helped object (green, high z) and an insignificant one (grey); bar chart of raw / ungated KD / ours at identical FPS |
| **Fig. 2** architecture (full width) | (a) Offline utility estimation: folds → out-of-fold teachers {R1, R2, E1–E3} → per-object ℓ → bin-wise R1−R2 MAD null → z_k(o). (b) Training: raw image → YOLOv8n student; frozen full-data teachers on raw and online E_k(x); P3–P5 features through "box mask × gate" into PKD feature and cls/DFL losses; raw-only background term; utility head on student RoIs gating pseudo-boxes from unlabelled images. (c) Inference: raw → student → boxes, "+0 FLOPs". Black data arrows, red dashed gradients, grey no-grad |
| **Fig. 3** utility distributions | (a) Log-y histograms of z_k per enhancer vs the R3 null with the τ = 2 line; (b) share of z > 2 by log-area and local luminance with bootstrap CIs |
| **Fig. 4** gate visualisation | 4–6 images, boxes coloured by argmax teacher, opacity ∝ g, enhanced thumbnails |
| **Fig. 5** scaling | (a) Δ vs training fraction for ungated and ours; (b) DarkFace mAP vs unlabelled amount; (c) training curves (supplement if short) |
| **Fig. 6** qualitative | 3 datasets × 2 images: raw, ours, ungated KD, AMIEOD-style; DarkFace tiny faces and NOD misses/false positives |
| **T1** ExDark-MAET main | All must-have baselines in our pipeline (mAP50, mAP50-95, AP_S, Δ vs raw with CI) + labelled published block |
| **T2** DarkFace + NOD (+ BDD-night) | Same columns; C3 variants |
| **T3** gate analysis | Gate types × {mAP50, on-rate, null FPR, reliability ρ}; ensemble-vs-enhancement decomposition row |
| **T4** ablations | Compact multi-block table |
| **T5** generality and efficiency | YOLOv8s student; RT-DETR-l student with RT-DETR-x teachers via custom encoder hooks (feature KD only); params/FLOPs/FPS/training GPU-h |
| Supplement | Null-scaling derivation, per-class and ten-condition results, YOLA-split ExDark, K = 5 and teacher-size ablations, TIDE, sweeps, all seeds, teacher P3 mAPs, gate failure cases, fold files, compute statement, SCI/Zero-DCE++ implementation notes |

The page budget is 8 pages, written to 7 of core content so an AAAI version needs no restructuring:

| Section | Pages |
|---|---|
| Introduction (computable-privilege insight, C1–C3, falsifiable predictions) | 1.0 |
| Related work | 0.75 |
| Method | 2.25 |
| Experiments | 3.5 |
| Limitations (training cost, teacher-quality dependence, the assumption that seed noise ≈ teacher-swap noise) and conclusion | 0.5 |

The page rules come from the 2025–2026 cycles. ICCV 2025 allowed eight pages plus references ([ICCV 2025 guidelines](https://iccv.thecvf.com/Conferences/2025/AuthorGuidelines)). ACM MM 2026 allowed 8 pages plus ≤ 2 reference-only pages and desk-rejects overlength papers ([ACM MM 2026 CFP](https://2026.acmmm.org/site/cfp-guidelines.html)). AAAI-26 allowed 7 pages plus references and a reproducibility checklist ([AAAI-26 instructions](https://aaai.org/conference/aaai/aaai-26/submission-instructions/)). The 2027 deadlines are projections:

| Venue | Projected deadline | Basis |
|---|---|---|
| ICCV 2027 (Hong Kong, 2–8 Oct 2027) | Full paper 5–8 March 2027, abstract about 1 March | [OpenCurious](https://www.opencurious.com/ai-conference-deadlines/iccv-2027) |
| ACM MM 2027 | Early April 2027 | Prior-year cycles |
| AAAI-28 | Late July to early August 2027 | Prior-year cycles |
| NeurIPS 2027 | — | Fits only a reframed, domain-general "cross-fitted privileged distillation" paper, which would be a different paper |

**ACM MM 2027 is the primary target.** It accepted C2FXNet's adverse-condition detection work in 2026, and its later deadline leaves time for three-seed tables. **ICCV 2027 is a go only if, by about 10 February 2027, all three gates have passed and ours beats the best ungated/AMIEOD-raw control by ≥ +1.0 on ExDark and ≥ +1.5 on DarkFace with paired CIs excluding zero.** NOD must point the same way. The original roadmap's stricter bar of ≥ +1.5 on all three datasets is what would make ICCV comfortable. Dual-submission rules forbid sending one paper to ICCV and MM at once, so an ICCV attempt that fails rolls to AAAI-28. TCSVT or TIP is the journal fallback.

### The three gates and their fallbacks

| Gate | When | Pass condition | If it fails |
|---|---|---|---|
| **GNG-1** utility is not noise | End of W3 (27 Nov) | For ≥ 1 enhancer, P(z > 2) ≥ 2× the R3-null rate with bootstrap CI excluding equality; ρ(z_E1, z_E1′) ≥ 0.3; z > 2 enriched in small/dark bins | Stop the method; write "per-object enhancement utility is indistinguishable from seed noise" into Paper 1 |
| **GNG-2** the gate beats its controls | End of W4 (4 Dec) | Ours − B8 ≥ +0.8 and ours − B11 ≥ +0.5, 3-seed mean, paired CI excluding 0 | If ungated enhanced-teacher KD (B8) beats multi-view (B5) by ≥ +1, reframe as "enhancement as computable privileged information" for AAAI/TCSVT; otherwise fold into Paper 1 |
| **GNG-3** C3 works | End of W7 (25 Dec) | ρ(ẑ, z) ≥ 0.4; recovery ≥ 70%; unlabelled data adds ≥ +1.0 mAP50 on DarkFace | Drop C3; submit C1 + C2 to ACM MM |

Two earlier checks catch failure sooner. In week 2, the raw-seed test–retest Spearman of ℓ should be at least about 0.5. In week 10, the gain at 25% data must exceed the gain at 100%, or the theory claim comes out.

## One GPU, two papers: a shared 20-week calendar

The two execution plans were costed separately, and they collide. **Paper 1 needs ≈1,220 GPU-hours (≈740 lean) and Paper 2 ≈650, or ≈1,870 together.** Paper 2's twelve weeks provide 2,016 wall-clock hours, so running both in that window would mean more than 90% utilisation of one card, which is unrealistic once crashes and debugging are counted. Three moves make the schedule work.

The first is a shared-run ledger. Harmonise the recipes (explicit SGD, 100 epochs, last epoch, the same image sizes and subset lists, augment-first for every enhanced YOLOv8m run), and these runs then serve both papers:

| Run | Serves Paper 1 | Serves Paper 2 | Saving (est.) |
|---|---|---|---|
| YOLOv8n raw, 3 datasets × 3 seeds | A1 | B1 | ≈15 GPU-h |
| YOLOv8m raw full-data, ExDark + DarkFace, seeds 0/1/2 | A10 raw arm | R1/R2/R3 full teachers | ≈25 |
| YOLOv8m SCI and Retinexformer augment-first, seed 0, ExDark + DarkFace | A10 enhanced arms | E3 and E2 full teachers | ≈17 |
| YOLOv8n P3 per enhancer | A4/A5 | B14 cascades | ≈25 |
| YOLOv8n raw at 10/25/50%, ExDark, same lists | A8 | Scaling raw arm | ≈5 |

Together these save roughly 80–90 GPU-hours. This is our estimate.

The second move is priority. Paper 2 has hard deadlines and Paper 1 has none, so Paper 2's gate-critical runs jump the queue, and Paper 1 runs its lean path inside the window. Paper 1's B-tier and the restoration of full seeds follow in February–March. If a second 24 GB GPU or about 500–600 cloud GPU-hours are available, give them to Paper 1's A4 (500 h) and run its full path.

The third move is to drive all jobs from one `runs.csv` queue with a `run_one.py --row k` runner. Keep `last.pt` and the prediction JSON for every run, and run CPU work (LIME, TIDE, bootstrap) in parallel.

| Week (Mon) | Paper 2 | Paper 1 | Decision |
|---|---|---|---|
| W1 (9 Nov) | Folds; YOLOv8m raw R1 full; `GatedDistillationModel` skeleton, `fuse()` test, B12 smoke test | Pre-flight fixes; data prep; `enhance.py`; `predict_dump.py`/`evaluate.py`/bootstrap; time one epoch per detector; A0 for core-4; YOLOv8n raw × 3 seeds (shared) | Teacher ≥ +3 over YOLOv8n, else YOLOv8l |
| W2 (16 Nov) | ExDark R1–R3 and E1 × 3 folds; ℓ extraction; pick ℓ definition | A1 (FRCNN, RT-DETR-l), A2 P0, A3 P2; rest of A0 | ℓ test–retest ρ ≥ 0.5 |
| W3 (23 Nov) | E2, E3, E1′ folds; full E1–E3 (shared with A10); z; Fig. 3 draft | P0/P2 heatmaps (Fig. 3 draft); start A4 YOLOv8n | **GNG-1** |
| W4 (30 Nov) | ExDark control grid B1–B12, B16, B17 at 1 seed, then 3 seeds for B8, B11, ours | A4 | **GNG-2** |
| W5 (7 Dec) | DarkFace teachers (1024 px) | A4 (FRCNN) | — |
| W6 (14 Dec) | DarkFace grid; ExDark ablations (C2 form, levels, τ, K) | A4 (RT-DETR-l, lean) | — |
| W7 (21 Dec) | C3 on the DarkFace unlabelled pool | A5 + A6 | **GNG-3** |
| W8 (28 Dec) | NOD teachers + reduced grid; BDD-night unlabelled | A7 | — |
| W9 (4 Jan) | YOLOv8s and RT-DETR students; SGLDet reimpl; official AMIEOD | A7, A8 | AMIEOD reproduces ≈ 81.9 ± 1 |
| W10 (11 Jan) | 3 seeds on all main rows; data scaling | A8, A9 | Prediction P1 holds |
| W11 (18 Jan) | Gate maps, size/luminance AP, decomposition, TIDE, efficiency, bootstrap + Holm; **re-run the novelty search** | A10 (remaining), A12 | — |
| W12 (25 Jan) | Final figures/tables; draft §3–4; code clean-up | A11 analyses; statistics; Figs 3–8 | — |
| W13–W14 (1–14 Feb) | Writing; **ICCV go/no-go about 10 Feb** | B1, B2, B4; lean-path seeds restored if GPU allows | Venue for Paper 2 |
| W15–W17 (15 Feb–7 Mar) | ICCV submission 5–8 Mar if go | Writing; B3, B5, B7–B10 | — |
| W18–W21 (8 Mar–4 Apr) | If no ICCV: ACM MM submission early April | Obtain Kumari et al. full text; re-check TMM/TCSVT early access; **submit to TMM late March–April** | — |

If Paper 2 goes to ICCV, Paper 1 follows in late March. If Paper 2 goes to MM, submitting Paper 1 first in mid-to-late March spreads the writing load.

**Pre-flight checklist (W1, before any run that will be reported).** SCI must save `r` (check `r ≥ x` on 20 images), PairLIE must save `I`, Zero-DCE++ must pad instead of cropping, Retinexformer must never use `--GT_mean`, and LIME must run with `-ul`; regenerate all affected outputs. Switch to the MAET 80/20 split and drop the old custom 70/20/10 split. Set the Ultralytics optimiser explicitly and never use `fraction`. Evaluate every detector with pycocotools on prediction dumps at conf ≥ 0.001 and max_det 300; Ultralytics' internal mAP interpolation differs, so it goes in the supplement only. Use three seeds everywhere a number will be compared.

**Before each submission**, re-run the arXiv and Xplore novelty search, read Kumari et al. (ASOC 2026) in full, re-read the M-AID and CA-MKD equations for Paper 2, and confirm current page limits, deadlines and dataset licences.

## Conclusion

The re-checks change the papers' centres of gravity, not their viability. Paper 1's value no longer lies in observing that enhancement sometimes helps; LoViF already shows that ExDark and DarkFace diverge. It lies in pricing that divergence: how much of the penalty is adaptable, how utility decays with data and capacity, and which pixel-level component moves which layer. Stratifying darkness within datasets and adding a synthetic ladder is what turns "protocol × dataset" into a defensible "protocol × darkness" claim. Paper 2's value has narrowed to a statistical idea: per-object enhancement utility is a small causal effect, and it must clear a seed-noise null before it is worth distilling. That idea is falsifiable within three weeks, at about 70 GPU-hours of ExDark teachers.

The most useful structural insight is that the two papers share experiments. The YOLOv8m capacity sweep is Paper 2's teacher set, and Paper 1's per-object regret is Paper 2's utility signal measured in-sample. A GNG-1 failure therefore does not waste work: it becomes a citable null result inside Paper 1. With one GPU, sequencing by deadline and sharing runs is what lets both submissions happen in spring 2027.
