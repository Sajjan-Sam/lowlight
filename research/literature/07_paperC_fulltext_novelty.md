# 07: Paper C ("Enhance for What, Not Where"): full-text novelty audit (2026-11-03)

**Method.** All papers below were read in full (arXiv HTML or CVF PDF), except TIP 2025, which is paywalled: repo and Crossref only, so its details are UNVERIFIED.

## 0. Verdict

- **The core observation is already in print, though unframed.** Morawski et al. ([IEEE TMM 2025](https://doi.org/10.1109/TMM.2025.3543047), [arXiv 2412.07693](https://arxiv.org/abs/2412.07693); CVPRW-NTIRE'24 version [2405.11478](https://arxiv.org/abs/2405.11478)) ran zero-shot YOLO-World and CLIP on ExDark under many enhancers. Every enhancer lowers detection mAP (40.3 → 31.7–40.2) and raises CLIP accuracy (22.8 → 33.7–47.7).
- **(ii)(b) and (iii) are textbook.** SIC-CADS (AAAI'24) and context rescoring (2009/10) cover them; an ExDark VLM-prior rescoring is on arXiv ([2509.19875](https://arxiv.org/abs/2509.19875)).
- **What survives:** a rigorous recognition-vs-localization decomposition across OV detectors and VLMs on real low-light data, plus an annotation artifact. The one technical delta is cross-view (raw + enhanced) presence consensus.
- **Novelty:** 4/10 as planned; 6/10 for the sharpest version (§5).

## 1. Assigned papers (full text)

| Paper | What it actually does | Relation to C |
|---|---|---|
| Language Prompt vs. Image Enhancement, [2604.10637](https://arxiv.org/abs/2604.10637) | Training-time **CLIP-CE**: CLIP ViT-B/32 on GT-box crops ("a photo without {cls}" vs. "of a {cls}") reweights Faster R-CNN's CE loss. No VLM or enhancer at test time. ExDark **5 classes**, 4411/2952 split, mAP50 68.59 → 70.42. Zero-DCE preprocessing alone gives 68.77, *above* the CE baseline. | No prior, no OV detection, no cross-view. 70.42 is not comparable to 12-class results. |
| VLOD-TTA, [2510.00458](https://arxiv.org/abs/2510.00458) (venue UNVERIFIED) | One-step TTA of YOLO-World-S and GDINO-T: IoU-weighted entropy plus selection of the top 25% of 16 GPT prompt **templates per class**. Zero-shot ExDark: YW 35.2 mAP / 64.7 AP50 / 34.6 AP75; GD 35.4 / 66.2 / 34.5. TTA gains +1.2 / +1.9 mAP. Also BDD100K. | No enhancers, no size analysis. Selects prompt wording, not classes. Use as the zero-shot baseline. |
| GEFU/SCUF ([ICCV'25](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_From_Enhancement_to_Understanding_Build_a_Generalized_Bridge_for_Low-light_ICCV_2025_paper.html)) | SD-Turbo cycle enhancer; a BLIP caption drives a *training* consistency loss. Frozen normal-light models: CODaN top-1 53.2 → 60.9 (Zero-DCE++ 59.9, LIME **14.1**); DarkFace mAP50 10.8 → 16.9. | Captions shape the enhancer only. Shows enhancement helps *frozen* classifiers. |
| DarkQA, [2512.24985](https://arxiv.org/abs/2512.24985) (RA-L'26 per [project page](https://darkqa-benchmark.github.io); arXiv: "submitted") | Synthetic indoor scenes, 9.4K MCQ, 5 EV-drop levels with RAW noise, 7 VLMs × 4 LLIE. **LLIE helps at L4–L5, hurts at L1–L3.** At L5 presence: FP 4.96%, **FN 89.36%**. | No boxes. Darkness causes *omission*, not hallucination. |
| EgoNight ([ICLR'26](https://proceedings.iclr.cc/paper_files/paper/2026/hash/022ca1bed6b574b962c48a2856eb207b-Abstract-Conference.html)) | Day/night egocentric video VQA (3,658 QA); GPT-4.1 best at 30.9%. | No enhancers, no boxes. |
| CLIP-LIT ([2303.17569](https://arxiv.org/abs/2303.17569)); PODA ([2212.03241](https://arxiv.org/abs/2212.03241)), both ICCV'23 | CLIP-loss backlit enhancement, no detection. Text-driven feature stylisation for zero-shot DA (ACDC-night 18.3 → 25.0 mIoU). | Lineage only. |
| TIP'25, He et al. ([DOI](https://doi.org/10.1109/TIP.2025.3641316), [repo](https://github.com/GingerCohle/VLMCSHFG)) | Day→night UDA on FCOS/SIGMA (SHIFT, BDD100K, FLIR): CLIP "hallucination feature generator" plus consistency sampler. **Full text UNVERIFIED.** | Training-time UDA; not ExDark. |
| NeurIPS'25 TTAOD ([proc.](https://proceedings.neurips.cc/paper_files/paper/2025/hash/88c3c482430a62d35e03926a22e4b67e-Abstract-Conference.html)) | GDINO multimodal-prompt mean teacher with DINOv2 memory rescoring. Benchmarks: Pascal-C, COCO-C, ODinW-13. | **No real low-light data.** |
| [2405.03519](https://arxiv.org/abs/2405.03519) (PBDL'24 challenge report) | Three CO-DETRs (trained on dark, IAT-enhanced, nuScenes-augmented data) plus HSV/resize TTA; GIoU clustering keeps the max-score box: 0.754 vs. 0.745. The IAT-trained model beat the dark-trained one (0.745 vs. 0.732). | Engineering only. |

## 2. Fresh 2025–26 hits

- **Morawski (TMM'25), Table V.** Zero-shot YOLO-World; CLIP classification per the CVPRW implementation details.
  - Detection / classification: raw 40.3 / 22.8; HE 33.3 / 39.9; LIME 38.3 / 38.9; Zero-DCE 34.0 / 44.1; SCI 40.2 / 35.3.
  - Table VIII (public checkpoints): PairLIE 38.0 / **47.7**. All 16 enhancer rows show the same split.
  - On the much darker LOD, enhancement *raises* AP50 and AP75 (SCI: AP75 41.5 → 50.2). **The effect depends on severity.**
- **[2509.19875](https://arxiv.org/abs/2509.19875).** A LoRA-tuned Qwen2-VL-7B emits a **category prior P(c)**, brightness and ROIs. These reweight YOLOv12s class scores and lower its threshold. ExDark mAP50: 0.775 → 0.832.
- **SIC-CADS ([2312.10439](https://arxiv.org/abs/2312.10439), AAAI'24).** A multi-label head on the detector's ResNet50-FPN features, L1-distilled from the CLIP image embedding. Fused as p_mlr^γ · p_det^(1−γ): +3.5–4.2 AP_r; with RAM or BLIP as the prior, +5.7–6.3.
- **Cross-view selection and aggregation:** FOCAL ([2507.10375](https://arxiv.org/abs/2507.10375), ICML'25: picks the CLIP-likeliest of gamma, colour and day/night views); TECA ([2403.17423](https://arxiv.org/abs/2403.17423): raw/enhanced logit switching); learned TTA aggregation ([2011.11156](https://arxiv.org/abs/2011.11156), ICCV'21).
- **Other evidence:**
  - [2607.22034](https://arxiv.org/abs/2607.22034): under severe underexposure, Qwen2-VL-2B accuracy falls 0.99 → 0.22 while verbalised confidence stays flat. Use token probabilities as reliability weights.
  - [2409.14461](https://arxiv.org/abs/2409.14461): frozen classifiers gain from LLIE; retrained ExDark detectors do not.
  - [2405.14874](https://arxiv.org/abs/2405.14874): OV-detector robustness, synthetic shifts only.
- **Canonical "old trick" citations:**
  - Classification rescoring: Harzallah et al. ([ICCV'09](https://hal.archives-ouvertes.fr/inria-00439516)), DPM ([TPAMI'10](https://doi.org/10.1109/TPAMI.2009.167)), Divvala et al. ([CVPR'09](https://doi.org/10.1109/CVPR.2009.5206532)).
  - Hard false positives are mostly classification errors: DCR ([ECCV'18](https://arxiv.org/abs/1803.06799)). Contextual rescoring: Pato et al. ([CVPR'20](https://arxiv.org/abs/1912.12290)).
  - Tag-then-ground: RAM ([2306.03514](https://arxiv.org/abs/2306.03514)) with Grounded-SAM ([2401.14159](https://arxiv.org/abs/2401.14159)); DetGPT ([2305.14167](https://arxiv.org/abs/2305.14167)).
  - GDINO false positives on absent prompts: [2406.19057](https://arxiv.org/abs/2406.19057). Hallucination metric: POPE ([2305.10355](https://arxiv.org/abs/2305.10355)).

## 3. Contribution status

| Contribution | Status | Closest work | Exact remaining difference |
|---|---|---|---|
| (i) OV detectors / VLMs × enhancers × real boxes × size | **PARTIAL** | Morawski (YOLO-World × ~16 enhancers × ExDark/NOD/LOD); VLOD-TTA (GD/YW on ExDark/BDD); DarkQA (synthetic VQA) | ≥4 OV detectors plus VLM grounders × 4 real datasets × size × IoU threshold, with a recognition/localization split. Breadth, not a new question. |
| (ii)(a) Prior selects OV prompts | **PARTIAL** (generic version DONE) | RAM→GDINO; DetGPT; VLOD-TTA (templates only) | The prior comes from raw+enhanced consensus in low light. Small delta. |
| (ii)(b) Prior rescales detector scores | **DONE** | Harzallah'09; DPM'10; SIC-CADS'24; 2509.19875 (ExDark) | Only the reliability-weighted cross-view consensus. Reviewers will say "TTA aggregation + SIC-CADS". |
| (iii) Distil into an image-level head | **DONE** (generic) | SIC-CADS visual MLR (a CLIP-distilled head on the detector backbone) | New only if a consensus teacher beats a raw-view CLIP teacher. Unproven. |

## 4. Is the core hypothesis already reported?

**Partly.**
- **Recognition up, detection down, on the same ExDark images (Morawski).** HE "33.3% det / 39.9% class" vs. raw "40.3% / 22.8%".
- **Frozen classifiers gain from enhancement.** GEFU: CODaN 53.24 → 59.88 with Zero-DCE++. 2409.14461: VGG 0.7415 → 0.8098 with UTV.
- **VQA is severity-dependent.** DarkQA: LLIE "can improve QA accuracy at some degradation levels, but is not uniformly beneficial".
- **Not found anywhere:** explicit framing as *localization* loss; AP50 vs. AP75 under enhancement on ExDark; multi-label presence metrics; a swap test.

**Blunt caveat.** Your evidence, mAP50-95 dropping 1.2–1.5× more than mAP50, is what almost any shift produces, and in P2 both metrics fall. It shows that localization is more fragile. It does not show that recognition improves.

## 5. Score, sharpest version, venue

**Novelty 4/10 as planned.** **Sharpest version, 6/10: "What enhancement buys and what it costs: a recognition/localization decomposition for detectors and VLMs in the dark".**
1. **Decomposition protocol on the same images:**
   - presence mAP/F1 and POPE-style hallucination;
   - AP at IoU 0.1 / 0.5 / 0.75;
   - TIDE Cls/Loc/Miss;
   - a **swap test**: raw boxes scored with enhanced-view class evidence, and the reverse.
2. **Coverage:** closed-set (P0, raw-trained) and OV detectors and VLMs; 4 real datasets; size; lighting condition; severity. Test Morawski's LOD-vs-ExDark sign flip.
3. **The artifact** below.
4. **One honest baseline,** "where from raw, what from consensus", compared against SIC-CADS (raw-only prior) and 2509.19875. Drop (iii) unless it beats SIC-CADS.

**Venue.** NeurIPS 2027 Evaluations & Datasets, only with the artifact (deadline ~May 2027, UNVERIFIED). Otherwise IEEE TMM (Morawski's venue) or TCSVT. ACM MM main only if GO-4 shows ≥ +2 mAP over SIC-CADS-style rescoring on ≥2 datasets, which is unlikely.

**Artifact for an E&D paper:**
- **(a) Presence labels.** Exhaustive open-vocabulary (~150 terms) multi-label presence labels for the ExDark test set plus DarkFace, BDD-night and LLVIP subsets. ExDark's incomplete labels make "hallucination" unmeasurable without them.
- **(b) Boxes.** Completion of missing boxes, plus per-box human-visibility labels from 2 annotators, with κ reported.
- **(c) Cached views.** Outputs of 8–10 enhancers plus identity and brightness-matched controls.
- **(d) Toolkit.** Unified OV/VLM runner, decomposition metrics, stratified reports, leaderboard.
- **Paperwork.** Datasheet, Croissant metadata, licence check (ExDark licence UNVERIFIED).

## 6. Pilot (≤1 week, one GPU; CLIP runs on CPU)

**Design.** Fix or exclude SCI first. ExDark test (737 images) × 10 views: raw, 7 enhancers, an identity control and a brightness-matched gamma control. Tune on the validation split (1,472).

**Sanity check (1 h).** Reproduce Morawski's CLIP top-1 on raw vs. Zero-DCE (≈23 → 44).

**Models.**
- CLIP ViT-B/32: per-class calibrated sigmoid.
- Florence-2-base ([2311.06242](https://arxiv.org/abs/2311.06242)): `<OD>` and caption → presence; `<OPEN_VOCABULARY_DETECTION>` for boxes.
- OWLv2-base-ensemble and YOLO-World-s v2: boxes, and presence from the max box score per class.
- Qwen2.5-VL-3B: yes/no **token probabilities**, 12 classes per image.
- YOLOv8n, both COCO-pretrained (P0) and raw-trained.

**Metrics** (paired bootstrap 95% CIs):
- multi-label presence mAP and macro-F1, recall at 5% FPR;
- hallucination P(present | absent) on a 150-image hand-audited subset;
- AP50 / AP75 / AP50-95 by S/M/L; TIDE.

**Experiments.**
- **E1:** presence vs. box metrics for each view.
- **E2:** the swap test.
- **E3:** the consensus prior, compared with the best single view, mean-TTA, FOCAL-style view selection and a raw-only prior.
- **E4:** YOLOv8n rescoring, and prompt pruning for YOLO-World and OWLv2.

**Go/no-go.**
- **GO-1:** ≥2 enhancers raise presence mAP by ≥ +2.0 (CI excludes 0) for ≥2 recognisers, while AP75 does not rise. **If this fails, kill Paper C** and fold the zero-shot rows into Paper A.
- **GO-2:** a blur or JPEG control matched on the mAP50 drop must give a ΔAP75/ΔAP50 ratio ≥0.3 lower than the enhancers. Otherwise "localization hurts more" is generic.
- **GO-3:** consensus presence mAP is ≥ +1.0 over the best single view and ≥ +3.0 over raw.
- **GO-4:** all three must hold:
  - rescored YOLOv8n gains ≥ +1.0 mAP50 over raw (CI excludes 0);
  - it gains ≥ +0.5 over raw-only-prior rescoring;
  - prompt pruning cuts false positives per image by ≥20% at equal recall.
- **Guard:** enhanced-view hallucination stays ≤ raw + 2 points.
- **Outcome:**
  - GO-1 and GO-2 only → benchmark/analysis paper (E&D or TMM).
  - All four → full Paper C.
  - GO-1 fails → stop.

## 7. Corrections to the roadmap and `03`

- **"No low-light box-level OV study"** is false: see Morawski and VLOD-TTA.
- **2604.10637** is training-time CLIP loss weighting on 5 classes, not text replacing enhancement at inference.
- **TIP'25** is day→night UDA on SHIFT, BDD and FLIR, not ExDark.
- **NeurIPS'25 TTAOD** uses no real low-light data.
- **"VLMs hallucinate in the dark" is backwards.** The dominant failure is missed objects (DarkQA: FN 89%).
- **Never claim "class-presence prior" or "distil into an image-level head" as new.**
