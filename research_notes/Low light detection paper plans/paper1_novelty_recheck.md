# Paper 1 novelty re-check (adversarial, literature to 3 Nov 2026): "When does low-light enhancement help detection?"

Method note: arXiv was queried directly through its API (title/abstract search, sorted by date, with every hit since May 2026 inspected and Sep–Nov 2026 listings read line by line). Full texts were downloaded for the LoViF CVPRW 2026 paper, LR-YOLO (HTML v3), the defogging benchmark (2510.03906) and DI-Retinex. Crossref, OpenAlex and web search were used for journals. **Not reachable from this session:** ScienceDirect, ResearchGate and MDPI pages (DNS blocked or stub responses); OpenAlex rate-limited (429). Kumari et al. (ASOC 2026), IAFE-YOLO+ (PR 2026) and the J. Imaging 2026 underwater paper could therefore not be re-read in full, and are marked UNVERIFIED where it matters. Search coverage of IEEE TIP/TCSVT/TITS/TMM and IJCV is via Crossref/OpenAlex/web search, not by browsing tables of contents. A Q1 paper that is not on arXiv and that these indexes do not surface could still exist.

## Q1. Is each of contributions (1)–(5) already published? (verdict, closest work, exact difference)

### Takeaway
No paper found (to 3 Nov 2026) does any of (2), (3) or (4) for low-light detection, or does the full (1) matrix. (1) and (5) are **PARTIAL**: every protocol cell and the size question have published precedents, and two 2026 papers may already report per-lighting-condition detection results. Verdicts: (1) PARTIAL (novel only as the integrated, CI-backed matrix); (2) NOVEL, with a real risk that the result is null; (3) NOVEL; (4) NOVEL for LLIE, with methodology precedents in other domains; (5) PARTIAL. Nothing is SCOOPED outright.

### Cited Findings

**Re-confirmed P0 (zero-shot) evidence, which governs the contribution (1) headline**
- LoViF CVPRW 2026 (Cheng, Xu, Pan) states that evaluation uses "frozen detectors pre-trained on normal-light images … instead of retraining them on the enhanced images". The detector is COCO YOLOv8, plus a face detector trained on COCO-WholeBody faces. — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/papers/Cheng_Generative_Low-Light_Enhancement_for_Computer_Vision_Revisiting_the_Gap_Between_CVPRW_2026_paper.pdf)
- LoViF Table 2 (mAP50), re-read from the PDF:

  | | ExDark | DarkFace |
  |---|---|---|
  | Raw ("Low") | 0.565 | 0.207 |
  | RetinexNet | 0.424 | 0.253 |
  | HVI-CIDNet | 0.469 | 0.261 |
  | GlobalDiff | 0.514 | 0.241 |
  | ZeroDCE | 0.548 | 0.260 |
  | FourierDiff | 0.574 | 0.256 |
  | AGLLDiff | 0.568 | 0.226 |
  | DSLR | 0.566 | 0.243 |
  | DBDiff (theirs) | 0.592 | 0.279 |

  - **ExDark:** only 3 of 14 baseline enhancers beat raw, all by at most +0.009.
  - **DarkFace:** all 14 beat raw (+0.019 to +0.055).

  This is a single run with no CIs, one detector per dataset, and P0 only. — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/papers/Cheng_Generative_Low-Light_Enhancement_for_Computer_Vision_Revisiting_the_Gap_Between_CVPRW_2026_paper.pdf)
- LIME-Eval (abstract re-read) argues that the "train detector on enhanced images" protocol is "prone to overfitting". It proposes frozen, normal-light-pretrained detectors with an energy-based confidence score instead. Earlier full-text audit numbers for YOLOX-m on ExDark:

  | Protocol | mAP |
  |---|---|
  | Raw | 49.2 |
  | P3, enhance-first | 45.9–47.6 |
  | P3, augment-first | 48.5–49.2 |
  | P0, raw | 35.0 |
  | P0, enhanced | 34.0–35.4 |

  — [arXiv 2410.08810](https://arxiv.org/abs/2410.08810)
- SID-NISM is an older P0 data point on ExDark: pretrained YOLOv2 without fine-tuning gets 31.54% raw, 33.39% with SID-NISM and 33.98% with LIME. This is a positive P0 effect of +2.4 on ExDark, so P0-on-ExDark is enhancer-dependent, not uniformly null. — [arXiv 2012.08707](https://arxiv.org/abs/2012.08707), via search snippet. NB the paper's "10 classes" vs ExDark's 12 is unexplained.
- Sun et al. NeurIPS 2022, from the earlier audit: YOLOv3 trained on VOC and tested on ExDark gives raw 46.27, SCI 48.47, ZeroDCE 41.01. — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/1cac8326ce3fbe79171db9754211530c-Paper-Conference.pdf)

**Re-confirmed P3 (enhanced→enhanced) evidence**
- Wu et al. 2409.14461 (abstract re-read): LLIE's effect on classification and detection "is inconsistent and can sometimes be harmful". It is still an 8-page arXiv preprint with no journal reference. — [arXiv 2409.14461](https://arxiv.org/abs/2409.14461)
- The underwater analogue retrains 7 detectors on outputs of 18 enhancers (126 models + 7 raw = 133 models), i.e. P3 only. It is published as IEEE J. Oceanic Engineering 49(2):606–621, Apr 2024. — [arXiv 2311.18814](https://arxiv.org/abs/2311.18814); [DOI 10.1109/JOE.2023.3302888](https://doi.org/10.1109/JOE.2023.3302888)
- AMIEOD (TMM), from the earlier audit, uses YOLOv3 on ExDark:

  | Training | Test on raw | Test on SCI |
  |---|---|---|
  | Raw | 76.5 | |
  | SCI-trained (P3) | | 76.6 |
  | Mixed (P4) | 76.9 | 76.2 |

  — [arXiv 2605.06084](https://arxiv.org/abs/2605.06084)

**New 2026 items found in this pass (none scoops a contribution)**
- *Machine-vision-oriented evaluation of LLIE algorithms* (SPIE PCAMV, 21 Sep 2026):
  - 4 enhancers (Zero-DCE, SCI, Retinexformer, LLFlow) with YOLOv5s on a "stratified ExDARK subset".
  - "The original low-light input achieves the highest mAP@0.5 and Recall."
  - One detector, one dataset, no protocol matrix. The training protocol and the stratification variable are not stated in the abstract (UNVERIFIED).

  — [DOI 10.1117/12.3125752](https://doi.org/10.1117/12.3125752)
- *Comparative Analysis of Image Enhancement Methods with YOLO26s for Object Detection Under Low-Light Conditions* (IDAP 2026, IEEE conference, Sep 2026). No abstract retrievable, so this is UNVERIFIED. By its title it is single-detector, and the venue is minor. — [DOI 10.1109/idap71869.2026.11728009](https://doi.org/10.1109/idap71869.2026.11728009)
- Kumari, Singh, Saurabh, *A systematic review of image enhancement and object detection in low-light scenarios with benchmark experiments*, Applied Soft Computing 2026 (Sep issue).
  - The abstract is not retrievable (ScienceDirect blocked; Crossref/OpenAlex carry none).
  - Search snippet: experiments with YOLOv8–v11 on ExDark, DarkFace and LLVIP; it states the effectiveness of enhancement for detection is "not yet fully understood".
  - **UNVERIFIED whether it crosses enhancers × detectors × protocols.** This is the highest-risk unread item for (1); get the full text before submission.

  — [DOI 10.1016/j.asoc.2026.115607](https://doi.org/10.1016/j.asoc.2026.115607)
- IAFE-YOLO+, *Illumination-adaptive feature enhancement for low-light object detection* (Pattern Recognition, Aug 2026). This is a method paper on ExDark, LLVIP, DARK FACE and VOC2007. A search snippet says that on YOLOv8n it "improves the Backlight mAP@50 from 0.352 to 0.392 and the Low-Light mAP@50 from 0.392 to 0.446". That is a **per-lighting-condition detection breakdown**. The condition taxonomy and dataset are UNVERIFIED: "Backlight" is not one of ExDark's 10 condition names. It evaluates the authors' own module, not an enhancer × protocol grid. — [DOI 10.1016/j.patcog.2026.113122](https://doi.org/10.1016/j.patcog.2026.113122); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320326000853)
- AegisBench (arXiv, 17 Oct 2026):
  - Three detector families (Faster R-CNN, YOLOv11, RT-DETR), 9 corruptions × 3 severities, three seeds.
  - A 4.5-stop exposure loss drives recall to 0.
  - Fine-tuning on milder low light recovers only 0.06–0.19 recall and "does not transfer to real night imagery".
  - No enhancers. This is a precedent for the seeds plus multi-family rigour that reviewers will expect.

  — [arXiv 2611.00077](https://arxiv.org/abs/2611.00077)
- *How Reliable Are Object Detection Models Beyond Clear Weather?* (arXiv, 15 Oct 2026). Source-only (P0-raw) evaluation of one-stage, two-stage and transformer detectors; the night drop reaches 62.90%. No enhancement. — [arXiv 2611.00071](https://arxiv.org/abs/2611.00071)
- *Raw Imagery Impacting Your AI* (arXiv, 29 Sep 2026). **Controlled, component-wise degradations** (SNR, MTF, GSD) on satellite vessel detection with YOLOv5s, YOLOX-S and NanoDet. "The impact of image quality depends on the degradation mechanism." This is a methodological precedent for (4)'s component logic, but it covers sensor degradations, not decomposition of enhancer outputs. — [arXiv 2609.38265](https://arxiv.org/abs/2609.38265)
- Defogging benchmark (arXiv 2510.03906, v2 Jan 2026). It "identifies when defogging is effective" on Foggy Cityscapes and ACDC, using filters, networks, chained pipelines and VLM editors. Detection is evaluated with pretrained detectors via FiftyOne; the text reviewed shows no detector retraining, so it appears to be P0 only (not fully verified). This is the closest "when does it help" framing in haze, without a protocol matrix, data scaling or interventions. — [arXiv 2510.03906](https://arxiv.org/abs/2510.03906)
- GeoSR-Bench (arXiv 2605.00310). For super-resolution: 9 SR models × 3 task models × 5 tasks (270 settings); "improvements in traditional SR metrics often do not correlate with gains in task performance". Again IQA ≠ utility, which is hygiene only for Paper 1. — [arXiv 2605.00310](https://arxiv.org/abs/2605.00310)
- CKA under controlled degradations already exists **in other domains**. arXiv listings include *Do 3D Medical Foundation Models See Through MRI Artifacts? A Controlled Study of Representation Robustness* (2608.06613) and *When Adaptation Hurts: Connecting Representational Drift to OOD Failures in MedSAM Fine-Tuning* (2608.21300); both match the "centered kernel alignment" + degradation/corruption query. Only titles were inspected (title-level evidence). No LLIE or detection CKA paper was found. — [arXiv 2608.06613](https://arxiv.org/abs/2608.06613); [arXiv 2608.21300](https://arxiv.org/abs/2608.21300)
- Review 2505.05759 (Liu & Fan; journal venue per search snippet is Neurocomputing 2025, UNVERIFIED). It reports that "zero-shot learning [enhancers] … showed consistently boosted performance across various vision tasks", while supervised enhancers gave "modest improvements". This is a literature-level synthesis, not a controlled protocol study. — [arXiv 2505.05759](https://arxiv.org/abs/2505.05759)
- Other 2026 low-light detection items on arXiv are methods or datasets, not utility studies:
  - Synthetic RAW low-light augmentation for fine-grained person-detection evaluation as a function of illumination — [2605.22455](https://arxiv.org/abs/2605.22455)
  - Lighting-aware contrastive pretraining evaluated on ExDark — [2606.06899](https://arxiv.org/abs/2606.06899)
  - C2FXNet — [2609.25693](https://arxiv.org/abs/2609.25693)

**Per-contribution verdict table**

| # | Contribution | Verdict | Closest works | Exact difference that keeps it open |
|---|---|---|---|---|
| 1 | Same enhancers × detector families under P0–P4 on ≥3 real datasets of different darkness, seeds + CIs, protocol × darkness interaction | **PARTIAL** (novel only as the integrated matrix) | LIME-Eval (3 protocols, 1 detector, 1 dataset); LoViF (P0, 14 enhancers, ExDark + DarkFace, 1 detector each); 2409.14461 (P3, 3 detectors, ExDark); AMIEOD (P4, 1 enhancer); Kumari ASOC 2026 (UNVERIFIED scope) | None of these has P2, more than one protocol on more than one dataset, multiple seeds, or CIs. The ExDark-vs-DarkFace contrast is **already visible in LoViF Table 2**, so the interaction is not a new *observation*. It is new only as a *tested, CI-backed statistical claim across protocols*. |
| 2 | Adaptable (P3−P2) vs residual (P1−P3) penalty, controlled for augmentation order, quantisation and I/O | **NOVEL** (no paper reports P2 next to P3 for detection) | LIME-Eval's augmentation-order and Bread-round controls; 2409.14461's "cannot add new information" remark | The concept is folklore. The operational decomposition with controls is unreported. **Risk:** LIME-Eval's augment-first P3 ≈ raw suggests the residual may be ≈ 0. |
| 3 | Utility vs training-set size (10–100%) and frozen vs fine-tuned backbone | **NOVEL** (no precedent found in low light, haze, underwater or SR) | JOE 2024 underwater: stronger backbones and longer schedules widen the raw–enhanced gap (earlier audit); AegisBench fine-tuning on milder levels | Nobody sweeps data fraction or freezing with enhanced inputs |
| 4 | Luminance-matched gain/gamma, colour, noise residual and detail interventions → layer-wise CKA/MMD → per-object ΔAP | **NOVEL for LLIE**; methodology PARTIAL | 2609.38265 (component degradations, satellite); 2608.06613/2608.21300 (CKA under artifacts or drift, medical); LIME-Eval synthetic sensitivity; LR-YOLO Jacobian analysis; Sun 2022 feature shift | Nobody decomposes *enhancer outputs* into components and links them to representation shift and per-object ΔAP |
| 5a | Real-data, multi-detector object-size effect | **PARTIAL** (open on real data) | LR-YOLO, synthetic COCO_Dark (re-verified: YOLOv8 AP_S 16.5 / AP_L 45.1 vs InstructIR→YOLOv8 AP_S 15.0 / AP_L 46.5) | Theirs is synthetic gamma darkening (γ∈[1.5,5]) with one restorer |
| 5b | ExDark 10 lighting-condition analysis | **PARTIAL, at higher risk than the roadmap states** | IAFE-YOLO+ PR 2026 (per-condition mAP for its own module; taxonomy UNVERIFIED); SPIE PCAMV 2026 ("stratified ExDARK subset", variable UNVERIFIED); NOD (extreme vs non-extreme lighting) | Claim "first **enhancer × protocol** analysis across ExDark's 10 conditions", not "first per-condition detection analysis" |

### Inferences
- **Overall novelty for a Q1 journal:** about 6/10, consistent with the roadmap. The paper is not scooped, but (1) on its own reads as a consolidation of LIME-Eval + LoViF + 2409.14461 + AMIEOD. **Acceptance will hinge on (2)–(4).**
- **Confound in contribution (1).** "Darkness" is confounded with dataset identity and task across ExDark (12 general classes, mixed lighting), DarkFace (one class, tiny faces) and NOD/BDD (driving). A reviewer will argue the interaction is *task/object-scale × protocol*, not *darkness × protocol*.
- **Mitigation for the confound:** stratify darkness *within* a dataset:
  - ExDark's 10 conditions
  - NOD's extreme-light instance tags
  - Image mean-luminance quantiles inside each dataset
  - A physics-based RAW-noise darkening ladder on a well-lit set (in the style of 2605.22455)

  Without this, the headline is vulnerable.
- Claim 5b should be narrowed now. The IAFE-YOLO+ snippet indicates per-condition reporting already appears in a 2026 PR paper.

### Gaps
- Full text of Kumari et al. ASOC 2026: whether it runs enhancers under several protocols is unknown.
- IAFE-YOLO+ condition taxonomy and dataset.
- SPIE PCAMV stratification variable and training protocol.
- IDAP 2026 YOLO26s comparison contents.
- IEEE Xplore early-access TCSVT/TIP/TITS papers from Oct–Nov 2026 that are not on arXiv could not be enumerated. This is the residual scoop risk.
- OpenReview review text for LIME-Eval: still not obtained.

## Q2. What the key prior works do NOT cover (re-read of abstracts and key tables)

### Takeaway
Each key paper covers one slice: one protocol, one detector, one dataset, or a non-LLIE domain. None reports P2, data-size scaling, controlled component interventions with representation metrics, or multi-seed CIs.

### Cited Findings

**Low-light detection studies**

- **LIME-Eval** ([arXiv 2410.08810](https://arxiv.org/abs/2410.08810); ICLR 2025 submission withdrawn, ratings 3/6/8/8 per the earlier audit).
  - Covers: P0 vs P3 critique; an energy-based label-free metric; human-preference benchmark.
  - Does not cover: P2; ≥2 detectors for the protocol claim; DarkFace or NOD; seeds or CIs; data scaling; interventions or CKA.
- **2409.14461** ([arXiv 2409.14461](https://arxiv.org/abs/2409.14461)).
  - Covers: 12 enhancers × 3 detectors on ExDark, P3 only (earlier audit); classification under clean-trained vs dark-trained regimes.
  - Does not cover: P0/P2/P4 for detection; other datasets; seeds or CIs; size or condition analysis; interventions. Not peer-reviewed as of Nov 2026 (no journal-ref on arXiv).
- **LoViF CVPRW 2026** ([CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/papers/Cheng_Generative_Low-Light_Enhancement_for_Computer_Vision_Revisiting_the_Gap_Between_CVPRW_2026_paper.pdf)).
  - Covers: P0 only, 14 enhancers, ExDark + DarkFace, one frozen detector each; a fusion method (DBDiff).
  - Does not cover: retraining protocols; seeds or CIs; analysis of *why* the datasets differ.

**Enhancement method papers with task evaluation**

- **Morawski et al. TMM 2025** ([arXiv 2412.07693](https://arxiv.org/abs/2412.07693)). The abstract confirms a CLIP-guided zero-reference enhancer and "task-based performance across many low-light datasets, including image classification, object and face detection", i.e. P0-style evaluation of their method against zero-reference baselines. The recognition-vs-localisation table (YOLO-World down, CLIP up) is from the earlier audit, not re-read here.
  - Does not cover: retraining protocols; scaling; interventions.
- **LR-YOLO, NeurIPS 2025** ([arXiv 2510.24232](https://arxiv.org/abs/2510.24232); [HTML v3](https://arxiv.org/html/2510.24232v3)).
  - Covers: a Lipschitz-mismatch mechanism. Training data is synthetic (COCO_Dark/VOC_Dark via gamma γ∈[1.5,5], 118,287 train images); ExDark (2,563 test images, 10 classes) is used only as out-of-domain test. The size table is on synthetic data.
  - Does not cover: real-data training protocols; P2; data scaling; CKA.

**Other domains**

- **"Is UIE All Detectors Need"** ([arXiv 2311.18814](https://arxiv.org/abs/2311.18814); IEEE JOE 2024).
  - Covers: P3 only (126 enhanced-trained + 7 raw-trained models).
  - Does not cover: P0/P2/P4; data fraction; frozen backbones; controlled interventions.
- **J. Imaging 2026 underwater revisit** ([DOI 10.3390/jimaging12010018](https://doi.org/10.3390/jimaging12010018); MDPI not reachable in this pass, so this rests on the earlier audit).
  - Covers: per-image mixed-set upper bound and quality-stratified effect.
  - Does not cover: low light; protocol matrix; scaling; CKA.
- **TaskGuard** ([arXiv 2609.08011](https://arxiv.org/abs/2609.08011)).
  - Covers: an abstract-confirmed post-hoc controller "for frozen restoration and detection pipelines". It is trained only on Gaussian degradation and transferred to motion blur, rain and defocus; DAWN natural rain. It cuts loss-negative interventions by 54.2% and keeps 98.8% of always-restore AP.
  - Does not cover: low light; retraining protocols; data scaling. Its regional counterfactual is the precedent for Paper 1's "regret rate"; cite it, do not claim it.

### Inferences
- The gap is consistent across all re-read works: **P2, training-data scaling, frozen vs fine-tuned, and intervention-to-representation links are absent everywhere.** These four items are the defensible core.
- **LoViF strengthens the "interaction" claim** (independent replication target). It also removes any chance of presenting the ExDark/DarkFace P0 contrast as a discovery.

### Gaps
- The Morawski TMM table and J. Imaging 2026 were not re-read in this pass (network). Their characterisation relies on the earlier full-text audit (`research/literature/06`, `07`).

## Q3. Expected results from published numbers; which outcomes are publishable; minimum result set for Q1

### Takeaway
Published numbers predict, in order of confidence:
1. **P2 ≪ P1** (a large covariate-shift penalty; not yet reported, but follows from P0 behaviour)
2. **P0 on DarkFace: consistently positive**
3. **P0 on ExDark: ≈ 0 with enhancer-dependent sign**
4. **P3 vs P1 on ExDark: 0 to −3 mAP**, shrinking toward 0 with augment-first order
5. **Utility falling as data and backbone capacity grow**

The main risk is that the residual penalty (P1−P3) is ≈ 0 under proper controls. That is publishable if framed as refuting the "information loss" narrative.

### Cited Findings

**Expected P0 effect sizes**
- **DarkFace, P0:**
  - LoViF: +0.019 to +0.055 mAP50 over raw 0.207, i.e. +9% to +27% relative, all 14 enhancers positive — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/papers/Cheng_Generative_Low-Light_Enhancement_for_Computer_Vision_Revisiting_the_Gap_Between_CVPRW_2026_paper.pdf)
  - TPAMI survey (earlier audit): DSFD AP50 0.195 → up to 0.268 — [arXiv 2104.10729](https://arxiv.org/abs/2104.10729)
- **ExDark, P0:**

  | Source | Detector | Raw | Enhanced |
  |---|---|---|---|
  | LoViF | YOLOv8 | 0.565 | 0.424–0.574 |
  | LIME-Eval | YOLOX | 35.0 | 34.0–35.4 |
  | Sun 2022 | YOLOv3 | 46.27 | SCI 48.47, ZeroDCE 41.01 |
  | SID-NISM | YOLOv2 | 31.54 | LIME 33.98 |

  Sources: [LoViF](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/papers/Cheng_Generative_Low-Light_Enhancement_for_Computer_Vision_Revisiting_the_Gap_Between_CVPRW_2026_paper.pdf); [LIME-Eval](https://arxiv.org/abs/2410.08810); [Sun 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/1cac8326ce3fbe79171db9754211530c-Paper-Conference.pdf); [SID-NISM](https://arxiv.org/abs/2012.08707)

**Expected P3 vs P1 gaps**
- ExDark, P3 vs P1:
  - LIME-Eval: −1.6 to −3.3 (enhance-first); −0.7 to 0.0 (augment-first)
  - 2409.14461: YOLOv7 76.09 raw vs 70.2–75.9
  - AMIEOD: SCI +0.1

  Sources: [LIME-Eval](https://arxiv.org/abs/2410.08810); [2409.14461](https://arxiv.org/abs/2409.14461); [AMIEOD](https://arxiv.org/abs/2605.06084)
- Underwater: raw is best for every one of the 7 detectors, and the gap widens with stronger backbones and longer schedules. This predicts contribution (3)'s trend: enhancement utility falls with capacity and training. — [arXiv 2311.18814](https://arxiv.org/abs/2311.18814)
- Low-light fine-tuning on milder synthetic levels recovers only 0.06–0.19 recall and does not transfer to real night (aerial SAR). This suggests that frozen/zero-shot regimes are where enhancement has room to help. — [arXiv 2611.00077](https://arxiv.org/abs/2611.00077)

**Expected size and component effects**
- Size, synthetic COCO_Dark (LR-YOLO): enhancement costs small objects −1.5 AP_S and helps large objects +1.4 AP_L — [arXiv HTML 2510.24232v3](https://arxiv.org/html/2510.24232v3)
- Degradation mechanism matters, and blur + noise combinations cost most (satellite, controlled). This supports the prior that noise and detail components, not luminance, drive harm. — [arXiv 2609.38265](https://arxiv.org/abs/2609.38265)

**Detection-aware methods on DarkFace** (large gains, but these are methods, not plain enhancement, so not directly comparable; numbers from the earlier internal audit `literature/05`, not re-verified in this pass)
- SGLDet: 59.7 → 76.6 — [ICLR 2026 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/2b00b3331bd0f5fbfdd966ac06338f6d-Paper-Conference.pdf)
- YOLA: 62.1 → 67.4, TOOD — [arXiv 2410.18398](https://arxiv.org/abs/2410.18398)

**Framing reference**
- V-information formalises why deterministic processing can *increase* usable information for a bounded learner. This is the correct theory frame if P3 > P1 anywhere. — [arXiv 2002.10689](https://arxiv.org/abs/2002.10689)

### Inferences

**Predicted pattern**

| Dataset | P0 Δ | P2 Δ vs P1 | P3 Δ vs P1 | Residual (P1−P3) |
|---|---|---|---|---|
| DarkFace | Positive (+2 to +5 AP50) | — | Unknown; plausibly ≈0 to slightly positive | — |
| ExDark | ≈ 0 ± 1, with large negatives for artefact-heavy enhancers (RetinexNet, HVI-CIDNet: −10 to −14) | — | — | 0–3 mAP enhance-first; ≈0–0.7 augment-first |
| All | — | Large negative (likely −5 to −20 mAP) | — | — |

The adaptable part (P3−P2) will dominate the penalty. The no-retraining P2 numbers are an inference from P0 behaviour, not a published number. No verified P1-vs-P3 numbers on DarkFace with *plain* enhancers were found (see Gaps). That cell is the most uncertain, and potentially the most interesting.

**Publishable outcomes** (each stands as a result if CIs are tight)
- A significant protocol × darkness interaction that survives within-dataset darkness strata. This is the strongest version.
- Residual ≈ 0 under augment-first and quantisation controls. Framed as "the enhancement penalty is almost entirely adaptable, so the 'information loss' narrative (2409.14461, LIME-Eval's IB remark) is wrong", this is a clean, citable correction.
- Monotone decline of enhancement utility with data fraction and with unfreezing, plus a **crossover point**: "enhancement helps below N images / with frozen backbones". This is practically useful and has no precedent.
- An intervention result where luminance-only reproduces most of the P0 gain and the noise residual explains most harm (with CKA shift predicting per-object ΔAP, R² reported with CIs). The direction is expected, but quantification is new.
- A real-data size effect, in either direction. Contradicting LR-YOLO is a bonus, not a requirement.

**Outcomes that weaken the paper**
- (a) All P1/P3/P4 differences inside CI overlap, *and* the scaling curve is flat. The paper collapses to "enhancement doesn't matter when you retrain", which is already known.
- (b) The interaction disappears under within-dataset darkness stratification. "Darkness" becomes "dataset/task", and the headline must be rewritten.
- (c) Interventions explain little variance (CKA does not predict ΔAP). The mechanistic part becomes descriptive.
- (d) Kumari ASOC 2026 or an unindexed TCSVT paper turns out to run a multi-protocol grid.

**Minimum result set for a Q1 journal (TMM / TCSVT / PR level)**
1. The full P0–P4 matrix:
   - ≥3 datasets, plus a within-dataset darkness stratification
   - ≥3 detector families
   - ≥5–6 enhancers spanning artefact levels, plus gamma and identity controls
   - 3 seeds, paired-bootstrap CIs, and a mixed-effects or ANOVA test of the protocol × darkness term
2. The P2/P3 decomposition with both augmentation orders and quantisation/identity controls.
3. At least **one** of claims (3) or (4) done fully, with a clear quantitative trend. Both is the version that makes it comfortable.

Size and the ExDark 10-condition analyses are supporting sections, not pillars.

### Gaps
- No verified P3-vs-P1 numbers with plain enhancers on DarkFace or NOD were found in this pass. DI-Retinex's DarkFace evaluation is frozen DSFD (P0), confirmed from its PDF ([arXiv 2404.03327](https://arxiv.org/abs/2404.03327)). One search snippet mentioned YOLOv5 trained from scratch on DarkFace with fixed enhancers as preprocessing, but the source paper was not identified or verified.
- No published P2 numbers exist for any dataset, so the magnitude prediction is inference only.

## Q4. Which Q1 venues have published closely similar analysis papers (for venue choice and citation)

### Takeaway
Pure "does restoration help recognition" analysis papers have been accepted at TPAMI (Pei et al. 2021, classification) and IEEE JOE (underwater detection, 2024). Low-light enhancement + detection *methods* with task analyses appear in TMM (Morawski 2025, AMIEOD), Pattern Recognition (IAFE-YOLO+ 2026) and ASOC (Kumari 2026 review + benchmark). For Paper 1, **IEEE TMM or TCSVT are the best fit**; Pattern Recognition is a solid alternative; TPAMI or IJCV only if all of (2)–(4) land strongly.

### Cited Findings
- Pei et al., *Effects of Image Degradation and Degradation Removal to CNN-Based Image Classification*, IEEE TPAMI 43(4):1239–1253, 2021.
  - Hazy, underwater, motion-blur and fish-eye images at 7 levels.
  - Degradation removal "could not be used to improve much the CNN-based classification performance".
  - Classification only; no data-size study.

  — [arXiv 1810.05552](https://arxiv.org/abs/1810.05552) (journal details via search listing)
- Wang, Guo, …, Chongyi Li, *Is Underwater Image Enhancement All Object Detectors Need?*, IEEE J. Oceanic Engineering 49(2):606–621, 2024. — [DOI 10.1109/JOE.2023.3302888](https://doi.org/10.1109/JOE.2023.3302888)
- Morawski et al., IEEE TMM (accepted 2024/25): an LLIE method evaluated by downstream classification and detection — [arXiv 2412.07693](https://arxiv.org/abs/2412.07693)
- AMIEOD (TMM per the earlier audit): a multi-expert enhancement selection paper with raw/enhanced/mixed training numbers — [arXiv 2605.06084](https://arxiv.org/abs/2605.06084)
- Pattern Recognition 2026, IAFE-YOLO+ (a method; per-lighting-condition results) — [DOI 10.1016/j.patcog.2026.113122](https://doi.org/10.1016/j.patcog.2026.113122)
- Applied Soft Computing 2026, Kumari et al. (review with benchmark experiments) — [DOI 10.1016/j.asoc.2026.115607](https://doi.org/10.1016/j.asoc.2026.115607)
- Knowledge-Based Systems 2025, *Advancing low-light image enhancement through deep learning: A comprehensive experimental study*. Contents UNVERIFIED (no abstract retrievable). — [DOI 10.1016/j.knosys.2025.113827](https://doi.org/10.1016/j.knosys.2025.113827)
- TPAMI 2022 low-light survey (Li et al.) with frozen-detector DarkFace evaluation — [arXiv 2104.10729](https://arxiv.org/abs/2104.10729)
- NeurIPS 2022, Sun et al., *Rethinking Image Restoration for Object Detection* (conference, method + analysis) — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2022/file/1cac8326ce3fbe79171db9754211530c-Paper-Conference.pdf)

### Inferences
- **TMM is the strongest venue fit.** It has published two directly adjacent LLIE-for-detection papers (Morawski, AMIEOD), so reviewers there will know the literature and value P2/scaling/intervention evidence.
- **TCSVT** is equally plausible; its reviewer pool overlaps.
- **Pattern Recognition** accepts empirical analyses but tends to want a method. Adding a simple derived recipe (e.g. a data-size-aware "enhance only when frozen or under N images" rule, or a regret-rate-gated baseline) would help there.
- **TPAMI precedent (Pei 2021) shows an analysis-only paper can land at the top tier,** but only with breadth across several degradations or a strong theory component. Paper 1 is low-light only, so TPAMI is a stretch.
- Avoid presenting it as a "systematic review/benchmark". ASOC/KBS already hold that framing (Kumari 2026, KBS 2025), and it invites "incremental" reviews.

### Gaps
- Not verified whether TCSVT, TIP or TITS published an LLIE-for-detection *analysis* paper in Oct–Nov 2026 (IEEE Xplore early access not browsable from this session).
- IJCV's *Benchmarking Low-Light Image Enhancement and Beyond* (Liu et al., 2021; includes face detection) is a known precedent from background knowledge but was **not verified** in this pass; confirm before citing.
