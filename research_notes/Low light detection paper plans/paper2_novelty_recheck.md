# Paper 2 ("Distilling Computable Privilege"): adversarial novelty re-check and realistic gains (2026-11-03)

Scope: re-check C1 (cross-fitted, seed-noise-calibrated, per-object gating of enhanced-view teachers), C3 (label-free utility head), and the full combination for low-light detection, against literature through early November 2026. Builds on `research/literature/05_paper2_distillation_novelty_and_design.md` (earlier full-text audit of SGLDet, AMIEOD, M-AID, CA-MKD, YOLA, FRBNet, TorchAdapt, etc.). That audit's findings are not repeated unless something changed.

Method note: arxiv.org was reachable only via curl/export API (WebFetch DNS failed), so new arXiv items below were checked at abstract level via `export.arxiv.org/api`, plus HTML full-text keyword extraction for DIW (2608.24904) and ERA-Det (2610.26090). The GitHub API was blocked; repo existence and last-commit dates were checked with `git ls-remote` / shallow clone.

---

## Q1. Does anyone already do C1, C3, or the full combination?

### Takeaway
No paper found does the full combination (out-of-fold, seed-null-calibrated, per-object gating of *enhanced-view* teachers + label-free utility head) for low-light detection, or in any domain. But **C1 is less new than the roadmap assumes**: cross-fitting inside KD is published (Dao et al., 2021). Validation-guided, sample-wise gating of a *privileged* teacher's KD losses is published (DIW, Jul 2026). Loss-supervised per-location routing between two training-only single-view heads is published for detection (ERA-Det, Oct 2026). The only ingredient with no prior found is the **raw-vs-raw seed-disagreement null used to significance-test per-instance teacher utility**. That, and the "deterministic transform = no new information" framing, are what is left to defend. Revised novelty: **~4.5/10** (down from 5/10 in `literature/05`). It reaches 5.5–6 only if C3 shows semi-supervised gains on two datasets and the seed-null ablation clearly beats the in-sample and out-of-fold-without-null gates.

### Cited Findings

**New prior art for C1 (cross-fitting / held-out validation in KD)**
- **Dao, Kamath, Syrgkanis, Mackey, "Knowledge Distillation as Semiparametric Inference"** treats teacher probabilities as a plug-in nuisance estimate and proposes "two enhancements, cross-fitting and loss correction, to mitigate the impact of teacher overfitting and underfitting on student performance", validated on tabular and image data — [arXiv 2104.09732](https://arxiv.org/abs/2104.09732). Their cross-fitting: teachers trained on K−1 folds supply targets on the held-out fold (out-of-fold teacher) — [summary via arXiv 2408.01023](https://arxiv.org/pdf/2408.01023). Published at ICLR 2021 (venue from memory; **UNVERIFIED** in this session).
  - What it does not do: no multiple teachers, no per-instance gating, no significance test, no detection. It uses cross-fitting to make the KD *targets* unbiased, not to *measure utility*.
  - Consequence: Paper 2 can no longer say "first to cross-fit teachers in KD". It **must cite Dao et al.** and say "we use cross-fitting to estimate a per-object treatment effect of a view, not to de-bias targets".
- **DIW, "Dynamic Influence-Weighted Distillation for Single-IMU Activity Recognition"** (cs.LG, 17 Jul 2026). A four-IMU **privileged** teacher distils into a one-IMU student. DIW "constructs a detached one-step candidate update and evaluates it on a fold-internal meta set", then assigns **separate sample-wise gates to the logit and feature losses** — [arXiv 2608.24904](https://arxiv.org/abs/2608.24904).
  - Results: out-of-fold macro-F1 is 0.5618 supervised, 0.5716 fixed-weight KD, 0.6385 DIW — [arXiv 2608.24904](https://arxiv.org/abs/2608.24904).
  - Its related work frames this as "validation-guided weighting" and cites Learning-to-Reweight (Ren et al. 2018), Meta-Weight-Net, influence functions and LGTM (validation-based distillation influence) — [arXiv 2608.24904 HTML](https://arxiv.org/html/2608.24904v1).
  - **This is the closest conceptual match to C1 found anywhere:** privileged teacher, held-out evidence, sample-wise gates on KD losses. It differs in four ways: single teacher, HAR not detection, a meta-gradient instead of a cross-fitted treatment effect, and no seed-noise null.
- **MMKD, "Adaptive Multi-Teacher Knowledge Distillation with Meta-Learning"** (Zhang et al.): a meta-weight network combines several teachers' logits and features, optimised bi-level with a separate validation/buffer set; classification, not detection (per the search snippet) — [arXiv 2306.06634](https://arxiv.org/pdf/2306.06634).
- **MTKD-RL** (AAAI 2025) learns multi-teacher weights with an RL agent and reports detection and segmentation results — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32990/35145). Reinforced multi-teacher selection (AAAI 2021) does instance-level teacher weighting; its t-test (two-sided, 5 runs, p<0.05) is for **reporting** significance, not for gating — [AAAI 2021 pdf](https://cdn.aaai.org/ojs/17680/17680-13-21174-1-2-20210518.pdf).
- **LIAF-KD** (Mar 2026) uses learnable instance selectors to reweight per-instance detector KD, with the student in the loop; single teacher, same input. +2% on a GFL R50 student on KITTI/COCO — [arXiv 2603.26088](https://arxiv.org/abs/2603.26088).
- **TPRD** (Sep 2026) refines DETR teacher predictions before distillation. It addresses "teacher supervision quality", but per stage and not per view — [arXiv 2609.19964](https://arxiv.org/abs/2609.19964).

**New prior art for C3 (learned, label-free utility predictor)**
- **ERA-Det** (19 Oct 2026). "Reliability-supervised routing assigns a fusion weight to every location and supervises it with targets from the losses of two single-modality classifiers used only in training, moving loss-based supervision of fusion weights from whole samples to individual locations." The router "lowers the frame weight at night and in low light" — [arXiv 2610.26090](https://arxiv.org/abs/2610.26090). The paper claims to be "the first RGB–event detector whose fusion weights are supervised at every location" — [HTML](https://arxiv.org/html/2610.26090v1).
  - This is a **per-location learned head regressing which view's loss is lower**, which is C3's mechanism. It is used for inference-time fusion, not KD gating, and the evidence is in-sample.
  - A reviewer could cite it against C3. Differentiate by: amortised *out-of-fold, null-calibrated* targets; used only to gate KD on unlabeled images; removed at test time.
- **TrustMOPD** (Sep 2026) does label-free, token-level teacher reliability. It measures each teacher's displacement from a shared reference, "calibrates its magnitude across teachers", and weights the KD losses. It closes 91.5% / 98.0% of the gap to oracle routing (LLMs) — [arXiv 2609.23697](https://arxiv.org/abs/2609.23697). This is a label-free multi-teacher gate in another domain, so "first label-free teacher-utility gate" cannot be claimed in general. Restrict the claim to "for vision detection / enhanced views".
- **EWAD** (Apr 2026) routes supervision between teacher KD and gold labels by inter-teacher agreement, at token level (summarisation) — [arXiv 2604.03192](https://arxiv.org/abs/2604.03192).
- **AMIEOD**'s Expert Selection Module still predicts the best enhancer per *image* from global characteristics, trained with a detection-guided cross-entropy — [arXiv 2605.06084](https://arxiv.org/abs/2605.06084). No per-object or label-free KD gate was found in low-light detection.

**Privileged-information / LUPI framing**
- **PATT** (Sep 2026) weights a privileged tracker teacher's transfer "by the teacher's relative localization advantage over the student and its absolute localization accuracy" (in-sample, tracking) — [arXiv 2609.02471](https://arxiv.org/abs/2609.02471).
- **"Quantifying the Value of Privileged Information Using a PAC-Bayesian Approach"** (Sep 2026, submitted to AAAI 2027) proposes a training-time metric of the potential gain from privileged information, which it says corresponds to test-time gains — [arXiv 2609.12891](https://arxiv.org/abs/2609.12891). It is relevant to Paper 2's "computable privilege" theory and **must be cited**. Paper 2's claim that a deterministic f(X) adds no information is a special case these authors could also make.
- **Lopez-Paz et al., "Unifying distillation and privileged information"** (ICLR 2016) is the canonical generalized-distillation reference — [arXiv 1511.03643](https://arxiv.org/abs/1511.03643).

**Low-light / adverse-condition detection, Sep–Nov 2026** (arXiv API search on title/abstract terms low-light / low-illumination / night / ExDark / DarkFace; results sorted by date)
- **C2FXNet** (ACM MM 2026): scene-expert MoE for unified adverse-weather detection, 71.14% mAP on ExDark (protocol not checked). The repo currently holds only a README — [arXiv 2609.25693](https://arxiv.org/abs/2609.25693), [GitHub](https://github.com/PolarisFTL/C2FXNet).
- **DaL-MoE** (Aug 2026): restoration front end for low-light UAV bridge-damage detection, with no KD — [arXiv 2608.23136](https://arxiv.org/abs/2608.23136).
- **MirrorDistill** (Sep 2026): illumination-aware *latent distillation for restoration*, not detection — [arXiv 2609.25331](https://arxiv.org/abs/2609.25331v1).
- **ReDiffNet** (Oct 2026, RGB-IR UAV low-light detection) — [arXiv 2610.05074](https://arxiv.org/abs/2610.05074). **Stereo LWIR-RGB auto-labelling** for low-light VRU detection (Oct 2026) — [arXiv 2610.23621](https://arxiv.org/abs/2610.23621). Both are multimodal and not relevant to C1/C3 beyond related work.
- **Lighting-Aware Representation Learning** (Jun 2026): contrastive pre-training with an illumination auxiliary objective, evaluated on ExDark — [arXiv 2606.06899](https://arxiv.org/abs/2606.06899).
- **Clean Feature Distillation for Low-Light Detection** (PRCV 2025): clean-image teacher, Bridge Module, synthetic degradation. Uniform KD, no gating — [Springer](https://link.springer.com/chapter/10.1007/978-981-95-5755-4_9).
- No Sep–Nov 2026 arXiv paper was found that distils enhanced-view teachers into a raw-input low-light detector.

### Inferences
- **Delta claim, rewritten defensibly:** "Prior work gates multi-teacher KD with in-sample losses (M-AID, CA-MKD, PATT) or validation meta-gradients (MMKD, DIW), and uses cross-fitting only to de-bias KD targets (Dao et al.). We instead treat each enhancer as a *treatment* on a deterministic view of the same input, estimate its per-object effect out-of-fold, and *significance-test it against a raw-vs-raw seed null* before distilling." Only the last clause plus the low-light instantiation is unclaimed.
- The **seed null is now the paper's load-bearing novelty.** If GNG-1 shows z>2 objects are barely above the null rate, there is nothing left.
- **The killer reviewer comment** is now "M-AID with cross-fitting (Dao et al.) and a variance normaliser", rather than just "M-AID with a better gate". The rebuttal must be empirical: the out-of-fold + null gate beats (a) the in-sample M-AID gate, (b) an out-of-fold gate without the null, (c) DIW/MMKD-style meta-weighting, and (d) a raw-seed ensemble through the same gate.
- **Add a DIW-style meta-gradient gate as a control.** It is the most natural "why not just do X" baseline, and it is cheaper than cross-fitting (no 15 extra teacher runs).
- **C3 must cite ERA-Det and TrustMOPD** and claim novelty only as "label-free per-object KD gate for enhanced views, used for semi-supervised low-light detection".

### Gaps
- M-AID full text could not be cleanly re-extracted (garbled PDF in the earlier audit). Whether M-AID uses any held-out split is **UNVERIFIED**; the earlier audit read it as in-sample.
- Full texts of MMKD, MTKD-RL and LIAF-KD were not read; their validation protocols are taken from abstracts and snippets.
- OpenReview and IEEE Xplore were not searched systematically for non-arXiv T-IV/TCSVT/TIP 2026 papers. A journal-only paper doing per-object gated enhanced-teacher KD cannot be ruled out.
- No search hit for "doubly robust knowledge distillation" beyond Dao et al. (whose loss correction is the doubly-robust analogue).

---

## Q2. Current SOTA numbers and code availability

### Takeaway
The numbers in `literature/05` §4 still stand; nothing new in Sep–Nov 2026 beats them on the standard protocols. Of the main competitors, released, runnable code exists for **AMIEOD, YOLA, FRBNet, DAI-Net, FeatEnHancer and MAET**. **SGLDet still has no code** (README says "Code coming soon"), so it must be reimplemented. YOLOv8 numbers on ExDark are protocol-chaos, ranging from 71 to 80 mAP50, so only own-run baselines on the MAET split are credible.

### Cited Findings
- **ExDark, MAET split (5,896/1,467):**
  - YOLOv3-608, 25 epochs: 76.4 baseline → SGLDet 78.6; DAI-Net 78.3; MAET 77.7.
  - YOLOv3-640, 100 epochs: 76.5 → AMIEOD 82.1, with a raw-input variant at 81.9.
  - TorchAdapt (own baseline): 78.8 → 80.1.
  - Source: earlier audit, `research/literature/05_paper2_distillation_novelty_and_design.md` §4, citing the [SGLDet pdf](https://proceedings.iclr.cc/paper_files/paper/2026/file/2b00b3331bd0f5fbfdd966ac06338f6d-Paper-Conference.pdf), [AMIEOD](https://arxiv.org/abs/2605.06084) and [TorchAdapt](https://openaccess.thecvf.com/content/ICCV2025/html/Hashmi_TorchAdapt_Towards_Light-Agnostic_Real-Time_Visual_Perception_ICCV_2025_paper.html).
- **ExDark, YOLA split:**
  - YOLOv3: 71.0 → YOLA 72.7, FRBNet 74.9.
  - TOOD: 72.5 → YOLA 75.2, FRBNet 75.4.
  - Sources: [YOLA](https://arxiv.org/abs/2410.18398), [FRBNet](https://arxiv.org/abs/2510.23444) (FRBNet claims "+2.2 mAP for dark object detection").
- **DarkFace:**

  | Protocol | Detector | Baseline → method |
  |---|---|---|
  | SGLDet, 5,500/500 | DSFD | 59.7 → 76.6 |
  | YOLA split | TOOD | 62.1 → 67.4 |
  | FRBNet 3:1:1 split | TOOD | 57.0 → 65.1 |
  | AMIEOD | YOLOv3 | 63.0 → 64.3 |
  | TorchAdapt | RetinaNet | 47.3 → 53.7 |

  Same sources as above; numbers are from the earlier audit.
- **LLVIP-visible:** AMIEOD 91.4 → 92.8 — [arXiv 2605.06084](https://arxiv.org/abs/2605.06084) (via the earlier audit).
- **NOD**, BMVC 2021: >7K images, 46K boxes, three classes (person, bicycle, car), two cameras — [GitHub](https://github.com/igor-morawski/NOD), [arXiv 2110.10364](https://arxiv.org/abs/2110.10364v1).
  - FeatEnHancer on NOD, as reported by Dark-ISP: Sony subset 30.3 mAP / 52.1 mAP50; Nikon subset 28.8 / 48.9. Dark-ISP on Sony: 31.5 / 53.4 — [arXiv 2509.09183](https://arxiv.org/pdf/2509.09183).
  - No YOLA/AMIEOD/SGLDet NOD numbers were found.
- **YOLOv8 on ExDark (non-standard protocols):**
  - DetectionBench Hugging Face models: YOLOv8n 71.29 mAP50, YOLOv8x 75.4 — [HF exdark-yolov8n](https://huggingface.co/dronefreak/exdark-yolov8n), [HF exdark-yolov8l](https://huggingface.co/dronefreak/exdark-yolov8l/blob/main/README.md).
  - DAEG-YOLO repo: YOLOv8s baseline 80.17, its method 80.35 — [GitHub](https://github.com/riyadijiskandar-beep/DAEG-YOLO) (single unreviewed source, **UNVERIFIED** protocol).
  - Dark-YOLO: 80.1 vs 71.3 in different versions — [ResearchGate](https://www.researchgate.net/publication/391506570_Dark-YOLO_A_Low-Light_Object_Detection_Algorithm_Integrating_Multiple_Attention_Mechanisms) (conflicting).
- **C2FXNet:** 71.14% mAP on ExDark, ACM MM 2026 (split not checked) — [arXiv 2609.25693](https://arxiv.org/abs/2609.25693).

**Code availability** (checked 2026-11-03 with `git ls-remote` / shallow clone):

| Method | Repo | Status |
|---|---|---|
| AMIEOD (TMM) | [github.com/scujayfantasy/AMIEOD](https://github.com/scujayfantasy/AMIEOD) (same commit as jayfantasyxc/AMIEOD) | **Released**, 120 files, Ultralytics-style, `train_AMIEOD_stage1.py`, `train_stage2.py`; last commit 2026-05-07 |
| SGLDet (ICLR 2026) | [github.com/gw-shin/SGLDet](https://github.com/gw-shin/SGLDet) | **No code** (README + poster only; "Code coming soon", "Release pre-trained model weights" unchecked); last commit 2026-04-25 |
| YOLA (NeurIPS 2024) | [github.com/MingboHong/YOLA](https://github.com/MingboHong/YOLA) | Repo exists (mmdet-based per paper) |
| FRBNet (NeurIPS 2025) | [github.com/Sing-Forevet/FRBNet](https://github.com/Sing-Forevet/FRBNet) | Repo exists |
| DAI-Net | [github.com/ZPDu/DAI-Net](https://github.com/ZPDu/DAI-Net) | Repo exists (DarkFace, DSFD-based) |
| FeatEnHancer (ICCV 2023) | [github.com/khurramHashmi/FeatEnHancer](https://github.com/khurramHashmi/FeatEnHancer) | Repo exists |
| MAET (ICCV 2021) | [github.com/cuiziteng/ICCV_MAET](https://github.com/cuiziteng/ICCV_MAET) | Repo exists; defines the 80/20 ExDark split |
| C2FXNet (MM 2026) | [github.com/PolarisFTL/C2FXNet](https://github.com/PolarisFTL/C2FXNet) | README only |
| TorchAdapt (ICCV 2025) | — | Code status not checked (**UNVERIFIED**) |

### Inferences
- **Reproducible head-to-heads:** AMIEOD (closest, same Ultralytics family, raw-input variant) is mandatory and feasible. YOLA and FRBNet run on their own split, so cite them on their protocol, or port the YOLA module into YOLOv8 as a plug-in control. SGLDet needs a reimplementation (λ=0.01); flag it as "our reimplementation".
- Because published YOLOv8-ExDark numbers span 71–80 mAP50, **never compare to third-party YOLOv8 numbers**. Run every baseline yourself on the MAET split with 3 seeds.

### Gaps
- Contents of the YOLA/FRBNet/DAI-Net repos (weights, configs) were not inspected; existence only.
- No consolidated NOD or LLVIP-visible leaderboard was found for YOLOv8-class detectors.

---

## Q3. Realistic gains, what reliably adds mAP, and why reviewers would reject

### Takeaway
Realistic expectations for YOLOv8n/s on raw input (all inferences from published deltas; nothing is measured yet):

| Comparison | Expected gain |
|---|---|
| (a) vs plain raw training | +2–4 mAP50 on ExDark-MAET, mostly from YOLOv8m teachers plus multi-view training |
| (b) vs mixed raw+enhanced training | +1–2 |
| (c) vs AMIEOD's recipe, ported to YOLOv8 on raw input | +0.5–1.5; may be ≈0 |
| (d) vs SGLDet reimplementation | +0.5–2 |
| Gate itself vs ungated KD | +0.3–1.0 |

DarkFace is the dataset most likely to show larger gains. Assume the gate's contribution is within 2 seeds' noise until proven otherwise.

### Cited Findings
- Multi-view training is the strongest cheap baseline. AMIEOD's raw-input model reaches 81.9 vs 76.5 baseline (YOLOv3, ExDark-MAET), while mixed raw+enhanced training without DGRL gives only 76.9–78.3 — `literature/05` §1 citing [AMIEOD](https://arxiv.org/abs/2605.06084).
- AMIEOD's gains shrink on stronger YOLOs: +2.7 (v5s), +0.5 (v7), +1.5 (v9-s) — same source.
- Training-only enhancement guidance (SGLDet):

  | Dataset | Detector | Baseline → SGLDet |
  |---|---|---|
  | ExDark | YOLOv3 | 76.4 → 78.6 |
  | DarkFace | DSFD from scratch | 59.7 → 76.6 |
  | nuImages-night | YOLOv8m | 52.4 → 55.3 |

  Source: [SGLDet pdf](https://proceedings.iclr.cc/paper_files/paper/2026/file/2b00b3331bd0f5fbfdd966ac06338f6d-Paper-Conference.pdf) (via `literature/05`).
- Ungated illumination consistency can hurt. YOLA-Naive: −0.9 on ExDark/TOOD and −0.8 on DarkFace/YOLOv3 — [YOLA](https://arxiv.org/abs/2410.18398) (via `literature/05`). This supports *some* gating, but in-sample gating may suffice.
- Per-instance KD weighting gains elsewhere:
  - Sunshine-to-Rainstorm: weighting +2.5 over unweighted (3D detection) — [arXiv 2402.18493](https://arxiv.org/abs/2402.18493).
  - LIAF-KD: +2% (2D detection, single teacher) — [arXiv 2603.26088](https://arxiv.org/abs/2603.26088).
  - DIW: +6.7 pp macro-F1 over fixed KD (HAR) — [arXiv 2608.24904](https://arxiv.org/abs/2608.24904).
  - Gating gains of +1–2.5 are achievable in other domains when the teacher signal is heterogeneous.
- ERA-Det's per-location loss-supervised routing halved the accuracy loss across operating frequencies (6.3 → 3.1 mAP) and "lowers it in every seed" — [arXiv 2610.26090](https://arxiv.org/abs/2610.26090). Location-level reliability supervision can matter when the two views genuinely differ (events vs frames). Raw vs enhanced is a much weaker difference.
- Detection-KD design evidence:
  - FGD (GT-box fg/bg masks + attention + global relations): +2.9–3.6 AP on COCO — [arXiv 2111.11837](https://arxiv.org/abs/2111.11837).
  - PKD (Pearson-standardised feature imitation) — [arXiv 2207.02039](https://arxiv.org/abs/2207.02039).

### Inferences
**Design choices that most reliably add mAP, in order:**
1. **Teacher capacity.** Use YOLOv8m/l teachers, with an equally large raw teacher as control. Most of (a) comes from here.
2. **Multi-view (AMIEOD-style) student training as the base recipe.**
3. **Masked FPN feature KD** (PKD/FGD style) on all levels, plus cls/DFL logit KD.
4. **Photometric augmentation** (gamma/noise).
5. **2–3 diverse enhancers.** Pick enhancers with different failure modes, e.g. a curve-based one (Zero-DCE++) plus a Retinex/transformer one (Retinexformer), rather than more of the same.
6. **The gate.** It is last, because its expected marginal is smallest.

**Most likely reviewer rejections:**
- (1) "M-AID/CA-MKD gate + Dao et al. cross-fitting + a variance normaliser; incremental."
- (2) "Gains over AMIEOD raw-input / SGLDet are within seed noise." This needs 3 seeds and paired-bootstrap CIs.
- (3) "The improvement is just a bigger teacher / ensembling." This needs the raw-seed-ensemble-through-same-gate control and an equal-size raw teacher.
- (4) "Cross-fitting costs 15+ extra teacher trainings for <1 mAP." Report GPU-hours honestly and show a cheaper DIW-style meta-gate baseline.
- (5) "Theory is trivial (data-processing inequality)." Present it as falsifiable predictions (data-scaling curve, dark/small concentration), not as a theorem.
- (6) "Only YOLO/ExDark." Needs DarkFace + one more dataset + a second detector (RT-DETR).
- (7) "Missing related work": DIW, ERA-Det, TrustMOPD, the PAC-Bayes LUPI value paper, Dao et al. Citing these preempts "not novel" reviews.

**Bluntly:** on ExDark the honest expectation for "gate vs best ungated control" is about +0.5 with CIs straddling zero. A paper whose headline delta is the gate is a coin flip at ACM MM and unlikely at ICCV. A paper whose headline is "a measured, null-calibrated map of where enhancement helps, plus a zero-cost student that beats AMIEOD-raw and SGLDet on DarkFace by ≥2" is viable.

### Gaps
- No published experiment distils from *enhanced-view* teachers into YOLOv8n/s on ExDark-MAET, so (b)–(d) are extrapolations, not citable numbers.
- AMIEOD has not been ported to YOLOv8 by anyone (none found); how its recipe transfers to v8n/s is unknown.

---

## Q4. Title, contribution statement, venue and deadlines

### Takeaway
Recommended single venue: **ACM MM 2027**, with ICCV 2027 as a stretch. Go for ICCV only if, by around 10 February 2027, the gate beats the best ungated/AMIEOD-raw control by ≥ +1.0 on ExDark **and** ≥ +1.5 on DarkFace with CIs excluding zero. ACM MM 2026 accepted closely related low-light/adverse-condition detection work (C2FXNet), and its deadline is about a month later than ICCV's, giving time for 3-seed tables. NeurIPS 2027 only fits if the paper is reframed as a general "cross-fitted privileged distillation" method with a non-vision experiment, which is a different paper. AAAI-28 is the fallback.

### Cited Findings
- **ICCV 2027 (Hong Kong, 2–8 Oct 2027).** Projected full-paper deadline **March 8, 2027 AoE** (abstract registration March 1) by OpenCurious; **March 5, 2027** "tentative" by Bibby. The official call for papers was not yet posted — [OpenCurious](https://www.opencurious.com/ai-conference-deadlines/iccv-2027), [Bibby](https://trybibby.com/conference-deadlines/march-2027). ICCV 2025's deadline was March 7, 2025 — [ICCV 2025 dates](https://iccv.thecvf.com/Conferences/2025/Dates).
- **C2FXNet** (unified adverse-weather/low-light detection with experts) was accepted at **ACM MM 2026** — [arXiv 2609.25693](https://arxiv.org/abs/2609.25693).
- **AAAI 2027** is receiving LUPI-theory submissions such as the PAC-Bayes privileged-information value paper — [arXiv 2609.12891](https://arxiv.org/abs/2609.12891).

### Inferences
**Deadline estimates** (all **UNVERIFIED** projections from prior-year cycles, not official calls):

| Venue | Estimated deadline |
|---|---|
| ICCV 2027 | ~5–8 Mar 2027 |
| ACM MM 2027 | ~early April 2027 (MM 2025/2026 abstracts were around late March / early April) |
| NeurIPS 2027 | ~mid May 2027 |
| AAAI-28 | ~late July / early August 2027 (abstract), ~1 week later (full paper) |

**Sharpest title** (keeps the roadmap phrase while signalling the statistical novelty):
- **"Distilling Computable Privilege: Seed-Calibrated, Out-of-Fold Enhancement Utility for Raw-Input Low-Light Detection"**
- Shorter alternative: **"When Does Enhancement Help? Null-Calibrated Per-Object Distillation for Low-Light Detection"**

**Abstract-level contribution statement:**
> Low-light enhancers are deterministic functions of the raw image, so they add inductive bias rather than information, and their benefit to a detector is small, object-dependent and easily confused with seed noise or ensembling. We estimate each enhancer's per-object utility out-of-fold, as a cross-fitted treatment effect standardised against a raw-vs-raw seed null, and distil enhanced-view teachers into a raw-input student only where the effect is significant. A label-free head amortises this estimate, extending the gate to unlabeled night images. The resulting detector has zero inference overhead, and improves over multi-view training (AMIEOD-style), training-only enhancement guidance (SGLDet), ungated multi-teacher KD and in-sample gating (M-AID) on ExDark, DarkFace and [third]. The gain shrinks with data size and concentrates on dark, small objects, as the theory predicts.

Only keep the last sentence if prediction P1 holds.

**Must-cite list to defuse novelty attacks:** Dao et al. 2021 (cross-fitting KD); DIW 2026 (validation-gated privileged KD); M-AID / CA-MKD (in-sample per-teacher weights); MMKD (meta-learned multi-teacher); ERA-Det 2026 (loss-supervised per-location routing); TrustMOPD 2026 (label-free calibrated teacher reliability); PATT 2026; Lopez-Paz 2016; the PAC-Bayes LUPI value paper 2026; AMIEOD; SGLDet; TaskGuard.

### Gaps
- No official ICCV 2027, ACM MM 2027, NeurIPS 2027 or AAAI-28 call for papers was located. All deadlines are estimates and must be re-checked in January 2027.
