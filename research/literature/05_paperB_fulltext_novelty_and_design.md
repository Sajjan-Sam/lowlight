# 05: Paper B, full-text novelty check and redesign (2026-11-03)

**Read in full:** SGLDet, AMIEOD, TaskGuard, Restore What Matters, FGD, CA-MKD, YOLA, FRBNet, FeatEnHancer, TorchAdapt, RMD-Net, KDFA, AID, Sunshine-to-Rainstorm, M-AID (garbled extraction; abstract, introduction and conclusion readable). **Abstract or method pages only:** ActLumos, MST-Distill, PATT, AMTML-KD, RL teacher selection, MFH, PKD, LUPI detection. **Blocked:** OpenReview, Springer, MDPI/PMC (bot challenges).

---

## 1. What the closest works actually do

| Work | What is transferred, and what is gated/weighted | Data / detector | Reported numbers |
|---|---|---|---|
| **SGLDet**, ICLR 2026 ([pdf](https://proceedings.iclr.cc/paper_files/paper/2026/file/2b00b3331bd0f5fbfdd966ac06338f6d-Paper-Conference.pdf)) | A U-Net decoder on the backbone, used in training only, reconstructs a target built from the phase of an SCI-enhanced image and the amplitude of an SDAP-denoised one. The loss is MSE with λ=0.01 (performance collapses at λ≥0.05). Dense and **ungated**. Code is "coming soon" ([repo](https://github.com/gw-shin/SGLDet)). | DarkFace (5,500/500 split), DSFD trained from scratch | 59.7→**76.6**. Enhanced-only target 72.9; gamma+Gaussian target 70.9. |
| | | ExDark, MAET split (5,896/1,467), YOLOv3-608, 25 epochs | 76.4→78.6 |
| | | nuImages-night, YOLOv8-m | 52.4→55.3 |
| **AMIEOD**, IEEE TMM ([arXiv 2605.06084](https://arxiv.org/abs/2605.06084), [code](https://github.com/jayfantasyxc/AMIEOD), Ultralytics) | Per **image**: b = argmin_k L_det(I_k, GT) over {**raw**, SCI, joint-SCI, IA-YOLO DIP}. Every expert's output is regressed to I_b (DGRL). The detector trains on all views. At inference a ResNet-50 selector is trained by cross-entropy to b. | ExDark MAET, YOLOv3 at 640, 100 epochs | 76.5→82.1 |
| | Ablation: **the same model fed raw images gets 81.9** at baseline FLOPs and FPS. Mixed raw+enhanced training without DGRL gives 76.9–78.3 on raw input. | DarkFace | 63.0→64.3 |
| | | LLVIP | 91.4→92.8 |
| | | Other YOLO versions | +2.7 (v5s), +0.5 (v7), +1.5 (v9-s) |
| **TorchAdapt**, ICCV 2025 ([CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Hashmi_TorchAdapt_Towards_Light-Agnostic_Real-Time_Visual_Perception_ICCV_2025_paper.html)) | Input enhancement + content-adaptive modulation; cosine alignment of enhanced-dark and well-lit features. Uniform; small inference cost. | ExDark, YOLOv3 | 78.8→80.1 |
| | | DarkFace, RetinaNet (AP50) | 47.3→53.7 |
| **RMD-Net**, IEEE TCyb ([pdf](https://dmas.lab.mcgill.ca/fung/pub/JHLFK23cyb_preprint.pdf)) | Daytime teacher; adversarial feature distillation on ground-truth RoIs. Uniform. | 16.5k day/night images; no ExDark | — |
| **KDFA**, Sensors 2025 ([doi](https://doi.org/10.3390/s25154871)) | Daylight teacher; RoI InfoNCE + geometric alignment on 1,200 synthetic pixel-aligned pairs. Uniform. | Apples | 51.3 mAP50:95 |
| **Clean Feature Distillation**, PRCV 2025 ([Springer](https://link.springer.com/chapter/10.1007/978-981-95-5755-4_9)) | Clean teacher → low-light student, Bridge Module, synthetic degradation. Title/venue confirmed via Crossref only. | — | **UNVERIFIED** |
| **TaskGuard** ([2609.08011](https://arxiv.org/abs/2609.08011)) | Per-image accept/reject of a restoration. Regional counterfactuals: 51.7% of regions helped, 48.3% hurt. **No low light.** | COCO, DAWN | Keeps 77.8% of DAWN AP gain |
| **Restore What Matters** ([2609.13791](https://arxiv.org/abs/2609.13791)) | Turbulence restoration + quality gate for face recognition. Not detection; cite for framing only. | BRIAR | — |
| **FGD**, CVPR 2022 ([2111.11837](https://arxiv.org/abs/2111.11837)) | Ground-truth-box foreground/background masks, a scale mask (inverse box area), teacher attention masks and global relations. Single teacher, same input. | COCO | +2.9 to +3.6 AP |
| **AID**, ICPR 2022 ([2201.11097](https://arxiv.org/abs/2201.11097)) and **M-AID**, IEEE T-IV 2022 ([pdf](https://par.nsf.gov/servlets/purl/10405166)) | Distillation weight **per instance, per scale and per teacher, computed from each teacher's ground-truth loss**. "The student learns the best knowledge from each teacher w.r.t. certain instances and scales." | KITTI, COCO-Traffic, SODA10M | +2.92 mAP on average |
| **CA-MKD**, ICASSP 2022 ([2201.00007](https://arxiv.org/abs/2201.00007)) | Per-sample teacher weights from each teacher's cross-entropy against ground truth. | CIFAR classification | — |
| AMTML-KD ([2103.04062](https://arxiv.org/abs/2103.04062)), RL teacher selection, AAAI 2021 ([pdf](https://cdn.aaai.org/ojs/17680/17680-13-21174-1-2-20210518.pdf)), **MST-Distill**, ACM MM 2025 ([2507.07015](https://arxiv.org/abs/2507.07015)) | Instance-level teacher weights or routing. MST-Distill routes across cross-modal teachers. | Classification / NLP | — |
| **PATT** ([2609.02471](https://arxiv.org/abs/2609.02471)) | Privileged teacher. Transfer is weighted by the **teacher's advantage over the student** and the teacher's absolute accuracy. | Tracking | — |
| **ActLumos** ([2502.03724](https://arxiv.org/abs/2502.03724)) | Teacher sees dark + Retinex frames through a learned per-timestep gate; the student sees dark frames only. | Action recognition | — |
| **Sunshine-to-Rainstorm**, AAAI 2024 ([2402.18493](https://arxiv.org/abs/2402.18493)) | Per-instance KD weights from the similarity between the sunny and rainy versions of each object. | 3D detection | Weighting adds +2.5 over unweighted |
| **YOLA** ([2410.18398](https://arxiv.org/abs/2410.18398)), YOLA-Naive variant | Uniform illumination-consistency MSE. **Ungated consistency can hurt.** | ExDark: TOOD / YOLOv3 | −0.9 / +0.6 |
| | | DarkFace: YOLOv3 / TOOD | −0.8 / +2.5 |

**No** detection paper gates distillation from *enhanced-view* teachers per object (searched: low-light, night, fog/rain, thermal→RGB, privileged-information detection [2601.02016](https://arxiv.org/abs/2601.02016)). Adverse-weather 2D work distils *uniformly*; the only per-instance cross-condition weighting is 3D (Sunshine-to-Rainstorm).

---

## 2. Verdict: Paper B as stated is **3/10**

Each part already exists: the gate is **M-AID** (+ CA-MKD normalisation); "lowest-GT-loss view, raw included" is **AMIEOD's DGRL** (per image); "training-only enhancement, zero inference cost" is **SGLDet** and AMIEOD's 81.9 raw-input result; "weight a privileged teacher by its advantage" is **PATT**; "utility differs within an image" is **TaskGuard**. A reviewer will write "M-AID applied to LLIE teachers".

**What is left:** (1) teachers differ only by a *deterministic transform of the same input*, so the "privileged" view carries no extra label information (I(Y; f(X)) ≤ I(Y; X)), unlike cross-modal KD ([MFH](https://arxiv.org/abs/2206.06487)); (2) per-object granularity for low light.

**Three defects in the current design:**
- **In-sample utility is noise.** P3 teachers saw the training images, so per-object losses are memorised; differences between two YOLOv8n runs are largely seed variance, so u_k(o) degenerates toward a random gate.
- **Ensemble confound.** Multi-teacher gains can come entirely from ensembling; no low-light KD paper controls for this.
- **Weak, same-capacity teachers.** Every P3 teacher is below raw and all are YOLOv8n, so KD headroom is small.

---

## 3. Redesign

**Title idea:** *Distilling Computable Privilege: Counterfactual Enhancement Utility for Raw-Input Low-Light Detection.*

**One-sentence contribution:** An enhancer is a deterministic function of the raw image, so it cannot add information, only inductive bias. We estimate that bias as a cross-fitted, seed-noise-corrected **per-object treatment effect**, distil enhanced-view teachers only where the effect is significant, and amortise the effect into a label-free head that extends the gate to unlabeled night images. The result is a plain raw-input detector with zero inference cost.

### Components (keep only what survives the literature check)

**C1. Counterfactual utility estimator (core novelty).**
- **Cross-fitting.** Split the training set into F=3 folds. For each fold, train these teachers on the other two folds:
  - a raw teacher with **two seeds**;
  - K=3 enhancer teachers: Zero-DCE++, KinD++, LIME. Drop SCI until it is fixed, and drop SNR-Aware (42% of its penalty is irreversible).
- **Utility.** For each object o, compute the out-of-fold losses ℓ, then
  - z_k(o) = (ℓ_raw(o) − ℓ_k(o)) / s(o), where s(o) is the robust scale of the raw-seed1 vs raw-seed2 disagreement, pooled within each (size × local-luminance) bin.
- **Gate.** g_k(o) = σ((z_k − τ)/T), with τ≈2. The raw teacher receives 1 − Σ_k g_k.
- M-AID, CA-MKD and AMIEOD use in-sample raw losses with no noise model; this estimator is the defensible delta.

**C2. Shift-invariant object transfer.**
- PKD-style (per-channel standardised) feature imitation inside each object's mask on its assigned FPN level, plus cls/DFL logit KD; background from the raw teacher.
- Standardisation strips the illumination-shifted feature statistics: "transfer structure, not the 58–90% recoverable shift". Not novel alone ([PKD](https://arxiv.org/abs/2207.02039)); present as a justified, ablated design choice.

**C3. Amortised utility head (the semi-supervised extension).**
- A small head on student object features regresses z_k(o); on unlabeled images (DarkFace unlabeled ≈9k, count UNVERIFIED; BDD100K-night) it gates KD on raw-teacher pseudo-boxes. Removed at test time (optionally exported as an "enhance here" map for Paper D).
- Closest: AMIEOD's selector and TaskGuard, both image-level, at inference. No per-object amortised KD gate found.

**Theory, as framing with falsifiable predictions rather than a claimed theorem:**
- P1: the gain from enhanced teachers shrinks as training-set size grows.
- P2: the gain concentrates where COCO-pretrained features differ most between raw and enhanced versions (dark, small objects).
- P3: a raw-seed ensemble with the *same* gate explains part of the gain; the remainder is enhancement-specific (a decomposition new to low-light KD).

**Drop as claims:**
- scale-conditioned gating (M-AID already does it);
- "per-teacher GT-verified weighting" (CA-MKD, M-AID);
- training-only enhancement (SGLDet).

**Novelty after the redesign: 5/10**, or 6/10 if the C3 semi-supervised gains hold on two datasets.
- Realistic targets: ACM MM 2027 / AAAI-28.
- CVPR/ICCV only if the method beats AMIEOD-style raw-inference training by at least +1.5 on three datasets, with confidence intervals.

---

## 4. Numbers that set the bar (all mAP50)

Protocols differ by split, resolution and number of epochs, so **never mix rows across protocols**.

| Protocol | Detector | Baseline → method |
|---|---|---|
| ExDark MAET split (5,896/1,467) | YOLOv3-608, 25 epochs | 76.4 → **SGLDet 78.6**, DAI-Net 78.3, MAET 77.7, SCI preprocessing 77.9 |
| ExDark MAET split | YOLOv3-640, 100 epochs (AMIEOD's setting) | 76.5 → AMIEOD 82.1. **Raw-input variant 81.9.** Mixed-view training without DGRL: 76.9–78.3 |
| ExDark (TorchAdapt's own baseline) | YOLOv3 | 78.8 → 80.1 |
| ExDark YOLA split (3,000/1,800/2,563) | YOLOv3 | 71.0 → YOLA 72.7, FRBNet 74.9 |
| ExDark YOLA split | TOOD | 72.5 → YOLA 75.2, FRBNet 75.4 |
| DarkFace (YOLA split 5,400/600) | YOLOv3 | 60.0 → 61.5 |
| DarkFace (YOLA split 5,400/600) | TOOD | 62.1 → 67.4 |
| DarkFace (FRBNet 3:1:1 split) | YOLOv3 | 54.8 → 57.7 |
| DarkFace (FRBNet 3:1:1 split) | TOOD | 57.0 → 65.1 |
| DarkFace (AMIEOD) | YOLOv3 | 63.0 → 64.3 |
| DarkFace (FeatEnHancer) | RetinaNet | 47.3 → 47.2 |
| DarkFace (FeatEnHancer) | FQ-RCNN | 67.5 → 69.0 |

**Realistic targets for YOLOv8n on the MAET split:**
- **vs no-KD raw:** +2–3 mAP50, but only with a larger teacher (YOLOv8m).
- **vs the strongest zero-cost control** (AMIEOD-style multi-view training with raw inference, plus same-teacher ungated KD): **+0.8–1.5 on ExDark** and **+1.5–3 on DarkFace**, where small faces are the objects enhancement helps.
- **Gate vs ungated:** probably only **+0.3–1.0**. This is the main risk.

**What reliably adds mAP, in order:** (1) teacher capacity: YOLOv8m/l teachers, controlled by an equally large raw teacher; (2) masked FPN feature KD (FGD/PKD) over logit-only, plus DFL/box KD; (3) all FPN levels, P3 carrying the small-object gain; (4) multi-view training of the student (AMIEOD: up to +5.4), **as the base recipe, not just a baseline**; (5) 2–3 diverse teachers, not more; (6) photometric augmentation.

**Mandatory controls** (same epochs and teacher size, 3 seeds, paired bootstrap 95% CIs on the 1,467-image MAET test set): no KD; extra-epoch fine-tuning; photometric augmentation; mixed-domain training (P4); AMIEOD raw-input; SGLDet reimplementation; single raw-teacher KD; ungated multi-teacher KD; random gate (matched gate-on rate); **in-sample M-AID/CA-MKD gate**; **raw-seed-ensemble teachers with the identical gate**; ours (ground-truth cross-fitted gate, label-free gate).

---

## 5. Ten-week plan (one 24 GB GPU, ≈550 GPU-hours)

Prerequisites, from roadmap §3: fix SCI, adopt the MAET split, and run 3 raw seeds.

| Week | Work | Go/no-go |
|---|---|---|
| 1 | Ultralytics KD hooks (PKD, cls/DFL, per-level masks); DarkFace 5,500/500; YOLOv8n raw ×3 seeds, YOLOv8m raw; AMIEOD raw-input recipe on YOLOv8n; photometric-aug baseline. | If AMIEOD-style multi-view training gives YOLOv8n ≥ +3 on raw input, it becomes the student's base recipe. |
| 2 | Train the cross-fitted YOLOv8m teachers: 3 folds × (2 raw seeds + 3 enhancers) = 15 runs, plus 5 full-data teachers. Compute z_k(o) and plot it against size and luminance. | **GNG-1:** if the share of objects with z_k > 2 is no larger than under a raw-vs-raw null (third seed), the signal is noise: fold the null result into Paper A and stop B. |
| 3 | Run the full control grid on ExDark, 1 seed, YOLOv8n. | **GNG-2:** cross-fitted gate ≥ +0.8 mAP50 over ungated KD **and** ≥ +0.5 over the raw-seed-ensemble gate (confirm on 2 more seeds). If not: if ungated enhanced-teacher KD beats multi-view training by ≥ +1, reframe as "enhancement as computable privileged information" (AAAI/TCSVT); otherwise move to Paper A/D. |
| 4 | Confirm seeds; DarkFace grid; ablations (PKD vs MSE, feature vs logit, FPN levels, K=1/2/3). | Gains must hold on DarkFace small faces (CI excludes 0). |
| 5 | Build the amortised utility head and semi-supervised KD on unlabeled DarkFace / BDD-night. | **GNG-3:** the label-free gate must recover ≥ 70% of the ground-truth gate's gain, and the unlabeled data must add ≥ +1.0 on DarkFace. Otherwise drop C3 and submit the supervised version to ACM MM. |
| 6 | Third dataset (nuImages-night, SGLDet's 3,591/894 split, or BDD100K-night); second detector (YOLOv8s or RT-DETR-R18). | — |
| 7 | Reimplement SGLDet (λ=0.01). Run AMIEOD officially on ExDark MAET. Run YOLA from its released code, or cite it on its own protocol. | — |
| 8 | Main tables with 3 seeds. Data-scaling curve (25/50/100%) to test prediction P1. | P1 must hold, or remove the theory claim. |
| 9 | Where the gate fires (size, luminance, class); TIDE; ensemble-vs-enhancement decomposition. | — |
| 10 | Write. Submit to ICCV 2027 (early March) only if GNG-2 and GNG-3 are passed with ≥ +1.5 over the best control on 3 datasets; otherwise ACM MM 2027 (~Apr 1). | — |

**Bottom line.** M-AID owns the gate. The paper is the claim that **enhancement utility is a small, noisy, object-level causal effect that must be estimated out-of-fold against a seed-noise null before it is worth distilling**. It must beat AMIEOD's 81.9 raw-input recipe and the raw-seed ensemble. If week 3 fails, stop.
