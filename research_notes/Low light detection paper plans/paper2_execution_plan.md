# Paper 2 execution plan: "Distilling Computable Privilege" (raw-input low-light detection with cross-fitted, noise-calibrated, per-object gated distillation)

Compiled 2026-11-03. Inputs: `research/PUBLICATION_ROADMAP.md` §1.2 and `research/literature/05_paper2_distillation_novelty_and_design.md` (cited below as "lit-05"; it is the full-text audit of SGLDet, AMIEOD, M-AID, CA-MKD, FGD, PKD, YOLA, FRBNet and others).

Verification method. Repositories were checked with `git ls-remote` / shallow clones on 2026-11-03, and code and configs were read directly:
- Ultralytics HEAD `73c60b7`, version 8.4.201
- MMRazor HEAD `90c7af1`
- FGD HEAD `c03a6cf`
- AMIEOD HEAD `120c6c9`
- SGLDet HEAD `6b7b81a`, last commit 2026-04-25
- YOLA HEAD `fe4f8c3`

`arxiv.org` and `par.nsf.gov` were **not reachable** from this session (DNS blocked), so paper-level numbers come from lit-05, which was read in full earlier, or from repo READMEs. Anything I could not confirm is marked **UNVERIFIED**. Design decisions and estimates are mine, and they sit under "Inferences".

---

## Q1. Implementation: adding gated teacher→student distillation to Ultralytics YOLOv8, with exact losses and default hyper-parameters

### Takeaway
Ultralytics now ships native detection distillation:
- a `DistillationModel` wrapper with forward hooks on the three neck layers that feed `Detect`;
- a 2×(1×1 conv) projector;
- a teacher-score-weighted L2 loss;
- args `distill_model=` and `dis=6.0`.

So the student should **subclass that wrapper**, not a third-party fork. Three things change in the subclass:
1. Each teacher gets its own (enhanced) input.
2. The L2 loss is replaced by box-masked, per-channel-standardised (PKD-style) feature KD, weighted per object by the gate.
3. A gated cls/DFL logit KD is added using the student's TAL assignment (`fg_mask`, `target_gt_idx`).

Default loss weights come from MMRazor/FGD configs: PKD weight 6–10 per FPN level; FGD temp 0.5 with α=1e-3 / β=5e-4 (one-stage) or 5e-5 / 2.5e-5 (two-stage). Sweep them, because YOLO feature scales differ from RetinaNet.

### Cited Findings
**Ultralytics native KD (the base to build on)**
- The current Ultralytics `main` has a Knowledge Distillation guide. Train with `model.train(data=..., distill_model="teacher.pt")`. `dis` (default **6.0**) is the distillation loss weight. The teacher stays frozen in `eval` mode. Features come from "the three neck layers that feed the Detect-family head". The projector is "two 1×1 convolutions with ReLU". The loss is a "score-weighted L2 loss … weighted by the teacher's classification confidence". — [Ultralytics KD guide](https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/knowledge-distillation.md); [default.yaml (`distill_model`, `dis: 6.0`)](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)
- Limits stated in the guide:
  - Only **detect** is experimentally verified.
  - "Cross-family distillation (e.g., YOLO11 teacher with YOLO26 student) is **not supported**".
  - "RT-DETR … not supported".
  - Published COCO gains are +0.4 to +1.0 mAP50-95, e.g. YOLO26n 40.9→41.5.

  — [Ultralytics KD guide](https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/knowledge-distillation.md)
- The feature was added in release **8.4.77**, PR #24894 ("Add knowledge distillation for YOLO detection models"). This comes from a third-party release tracker; the PR number was not opened directly. — [newreleases.io 8.4.77](https://newreleases.io/project/pypi/ultralytics/release/8.4.77)
- Code facts in `ultralytics/nn/distill_model.py`:
  - `get_distill_layers()` returns the Detect head's input layers plus the head index. Its docstring gives "YOLOv8 -> [15, 18, 21, 22]".
  - `FeatureHook` stores each layer output in a dict.
  - In `loss()`, the teacher runs `with torch.no_grad(): self.teacher_model(batch["img"])` on the **same** `batch["img"]` as the student.
  - Teacher scores are `sigmoid().max(dim=1)` over classes, split per level.
  - `loss_sl2` = Σ(MSE·score) / (Σscore·C).
  - The distillation loss is multiplied by `dis` and by batch size, then appended to the loss vector.
  - `fuse()` strips the wrapper, so the exported model is the plain student.

  — [distill_model.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py)
- The trainer wraps the model when `args.distill_model` is set (`trainer.py` L437–438). `torch.compile` is disabled under distillation (L379–380). — [trainer.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py)
- In training, the base `Detect.forward_head` returns a dict `{"boxes", "scores", "feats"}`. That is the format `decouple_outputs()` expects, so YOLOv8 (non-end2end) heads are structurally compatible. — [head.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/head.py)
- Ultralytics code is **AGPL-3.0** (file headers), so a released derivative must be AGPL-3.0. — [distill_model.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/distill_model.py)

**Per-object losses from a teacher (needed for C1)**
- `v8DetectionLoss` builds the assigner as `TaskAlignedAssigner(topk=tal_topk (default 10), alpha=0.5, beta=6.0)`.
- `get_assigned_targets_and_loss()` returns `(fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor)` together with the loss.
- BCE uses `reduction="none"`.
- Box loss is CIoU, plus DFL on `fg_mask` anchors.
- Default gains: `box: 7.5`, `cls: 0.5`, `dfl: 1.5`.
- Per-anchor losses can therefore be scattered to ground-truth objects through `target_gt_idx[fg_mask]`.

— [loss.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py); [default.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)
- YOLOv8 sizes from the model YAML:

  | Model | Params | GFLOPs (640) | Layers |
  |---|---|---|---|
  | YOLOv8n | 3,157,200 | 8.9 | 129 |
  | YOLOv8s | 11,166,560 | 28.8 | 129 |
  | YOLOv8m | 25,902,640 | 79.3 | 169 |
  | YOLOv8l | 43,691,520 | 165.7 | — |

  — [yolov8.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml)

**Community YOLOv8 KD forks (older alternative)**
- Chinese-language blog patches add `CWDLoss`/`MGDLoss` to Ultralytics by passing a teacher into the trainer. They are unofficial and tied to specific versions. DAMO-YOLO reported that CWD suited its models better than MGD. — [CSDN MGD-for-YOLOv8 blog](https://blog.csdn.net/qq_41335232/article/details/142717122); [DAMO-YOLO report](https://arxiv.org/pdf/2211.15444); [YOLOv8 pruning+CWD paper](https://arxiv.org/pdf/2509.12918)

**Reference losses and their official defaults**
- **PKD loss (MMRazor):**
  - `norm()` standardises each channel over (N,H,W): `(feat−mean)/(std+1e-6)`.
  - Loss = `F.mse_loss(norm_S, norm_T)/2` per level, times `loss_weight`.
  - The student is bilinearly resized if the spatial sizes differ.
  - Defaults: `loss_weight=6` per FPN level for Faster R-CNN R101→R50; `loss_weight=10` for RetinaNet X101→R50 (lr 0.01).
  - Reported COCO result: RetinaNet-R50 37.4→40.8 with a RetinaNet-X101 teacher.

  — [pkd_loss.py](https://github.com/open-mmlab/mmrazor/blob/main/mmrazor/models/losses/pkd_loss.py); [PKD config (FRCNN)](https://github.com/open-mmlab/mmrazor/blob/main/configs/distill/mmdet/pkd/pkd_fpn_faster-rcnn_r101_faster-rcnn_r50_2x_coco.py); [PKD config (Retina)](https://github.com/open-mmlab/mmrazor/blob/main/configs/distill/mmdet/pkd/pkd_fpn_retina_x101_retina_r50_2x_coco.py); [PKD README](https://github.com/open-mmlab/mmrazor/blob/main/configs/distill/mmdet/pkd/README.md)
- **CWD (MMRazor):** `ChannelWiseDivergence, tau=1, loss_weight=10` on each of 5 FPN levels. — [cwd_fpn_frcnn config](https://github.com/open-mmlab/mmrazor/blob/main/configs/distill/mmdet/cwd/cwd_fpn_frcnn_r101_frcnn_r50_1x_coco.py)
- **MGD (MMRazor):** `alpha_mgd=0.00002`, `lambda_mgd=0.65` per level. — [mgd config](https://github.com/open-mmlab/mmrazor/blob/main/configs/distill/mmdet/mgd/mgd_fpn_retina_x101_retina_r50_2x_coco.py)
- **FGD official configs.** All use `temp=0.5` except FCOS (0.8):

  | Detector family | α_fgd | β_fgd | γ_fgd | λ_fgd |
  |---|---|---|---|---|
  | RetinaNet / GFL / SOLO | 0.001 | 0.0005 | 0.0005 | 0.000005 |
  | Faster / Mask / Cascade R-CNN | 0.00005 | 0.000025 | 0.00005 | 0.0000005 |
  | FCOS / RepPoints | 0.0016 | 0.0008 | 0.0008 | 0.000008 |

  — [FGD configs](https://github.com/yzd-v/FGD/tree/master/configs/distillers/fgd)
- **FGD masks:**
  - Foreground mask value = max over the boxes covering a pixel of the scale weight `1/(h·w)`, with h·w the box size in feature-map pixels.
  - Background mask = 1 outside boxes, normalised to sum to 1.
  - Losses use `sqrt(mask)`-weighted features, then MSE divided by batch size.

  — [fgd.py L83–157](https://github.com/yzd-v/FGD/blob/master/mmdet/distillation/losses/fgd.py)
- **M-AID:** per-instance, per-scale, per-teacher KD weights computed from each teacher's ground-truth loss; +2.92 mAP on average across KITTI/COCO-Traffic/SODA10M. **CA-MKD:** per-sample teacher weights from each teacher's cross-entropy against ground truth. Exact equations and temperatures could **not** be re-read (arXiv and NSF blocked). — lit-05 table; [M-AID pdf](https://par.nsf.gov/servlets/purl/10405166); [CA-MKD arXiv](https://arxiv.org/abs/2201.00007)

### Inferences (the implementation recipe)
**Step 0. Environment.**
- Pin `ultralytics==8.4.201` or the commit `73c60b7` checked here. The KD API is <1 year old and still changing.
- Write a `GatedDistillationModel(DistillationModel)` and a `GatedDistillTrainer(DetectionTrainer)` in your own package. Do not edit site-packages.
- Unit-test `fuse()`: the exported student must have exactly 3,157,200 params for v8n.
- Verify experimentally that the native wrapper runs with a YOLOv8m→YOLOv8n pair. The docstring lists YOLOv8 layer indices, but the guide only benchmarks YOLO26. **UNVERIFIED for v8.**

**Step 1. Enhanced teacher inputs, kept geometrically aligned.**
- Recommended: apply the enhancer **online on the GPU** to the already-augmented raw batch, `x_k = E_k(batch["img"])`, inside `loss()` under `no_grad`. This keeps mosaic and affine transforms aligned automatically.
  - Zero-DCE++ and SCI are tiny.
  - Retinexformer (~1.6M params, **UNVERIFIED** count) is still cheap at 640.
- Offline alternative: cache enhanced images and load raw+enhanced as a 6-channel tensor through the augmentation pipeline. HSV augmentation would then have to be applied to raw only, so it needs a custom dataset (more code).
- Order of operations: Ultralytics' HSV jitter (`hsv_v: 0.4`) runs before the enhancer sees the image. That is "augment-first" in Paper 1's terms. State it, and ablate enhance-first (offline) once.

**Step 2. Teachers as a ModuleList.**
- `teachers = {raw_s1, raw_s2?, zdce, retinex, sci}`, all YOLOv8m, frozen, fp16.
- Register `FeatureHook` on layers [15,18,21,22] of each teacher.
- Cost control: for each batch, run enhanced teacher k only if some object in the batch has g_k > 0.05. Otherwise skip its forward pass.

**Step 3. Gated feature KD (C2).** For level l ∈ {P3, P4, P5} (strides 8/16/32):
- F̂_S = proj_l(F_S,l), using the 1×1–ReLU–1×1 projector already built by the wrapper.
- Standardise per channel over (N,H,W) as in PKD: N(F) = (F − μ_c)/(σ_c + 1e-6). Do this for F̂_S and for each teacher map F_t,l.
- For each object o:
  - box mask M_o,l on the level-l grid;
  - FGD scale weight a_o = 1/(h_o,l · w_o,l);
  - weights w_t(o) over teachers t ∈ {raw} ∪ {enh_k} (see Q2/C1).
- Loss:

  L_feat = Σ_l Σ_o a_o Σ_t w_t(o) · ‖M_o,l ⊙ (N(F̂_S) − N(F_t))‖² / C

  plus a background term λ_bg · ‖(1−M_fg) ⊙ (N(F̂_S) − N(F_raw))‖² / (C·|bg|), using the **raw** teacher only.
- Starting weights: λ_fg = **6** (PKD-FRCNN / Ultralytics `dis` default) and λ_bg = λ_fg/2 (FGD's α:β = 2:1).
- Sweep λ_fg ∈ {1, 3, 6, 10} on fold-0 validation.
- Level assignment: default "all levels for every object" (FGD masks all levels). Ablate "size-assigned level only".

**Step 4. Gated logit KD.** Use the **student's** TAL assignment (`fg_mask`, `target_gt_idx`) so every positive anchor maps to an object o.
- Classification: L_cls-KD = Σ_anchors w_t(o) · BCE(σ(s_S), σ(s_T)), with sigmoid targets and temperature T=1 (the head is sigmoid/BCE, not softmax).
- Boxes: L_dfl-KD = Σ w_t(o) · KL(softmax(d_T/τ_LD) ‖ softmax(d_S/τ_LD)) · τ_LD² over the 4×16 DFL bins.
  - τ_LD = 10 follows Localization Distillation (CVPR 2022). The value is from memory: **UNVERIFIED**, so sweep {1, 5, 10}.
- Starting weights λ_cls = 1, λ_dfl = 1 (my choice; sweep {0.5, 1, 2}).
- Background anchors: raw-teacher cls KD only, weight 0.25. This is a design choice to ablate.

**Step 5. Total loss.**

L = L_det(student, raw GT) + λ_fg·L_feat + λ_bg·L_bg + λ_cls·L_cls-KD + λ_dfl·L_dfl-KD [+ λ_u·L_util (C3)]

- Warm up the KD weights linearly over the first 3 epochs.
- Keep Ultralytics defaults otherwise: SGD lr0 0.01, mosaic 1.0, `close_mosaic: 10`, 100 epochs, imgsz 640.

**Step 6. Per-object utility (C1) is offline, not in the training loop.**
- Run each out-of-fold (OOF) teacher on its held-out fold in eval mode: letterbox 640, no augmentation, conf 0.001.
- Compute a per-object loss ℓ_t(o) in one of two ways:
  - (a) **Assignment-based:** feed the teacher's raw head outputs and that image's GT into `v8DetectionLoss.get_assigned_targets_and_loss` with `reduction="none"`, then sum box·7.5 + cls·0.5 + dfl·1.5 over anchors with `target_gt_idx == o`.
  - (b) **Assignment-free detection score:** ℓ_t(o) = −log(max over predictions p of class c_o of [score(p)·IoU(p,o)] + ε).
- Pick between (a) and (b) in week 2 by **test–retest reliability**, the Spearman correlation of ℓ between raw seed 1 and raw seed 2 (higher is better). (b) is more robust to TAL assignment differences between teachers.
- Store a Parquet table, one row per object: (img_id, obj_id, fold, class, area, local luminance, ℓ_raw1, ℓ_raw2, ℓ_raw3, ℓ_k…).

**Step 7. z-score and gate (C1).** Put each object into one of 9 bins: size tercile × local-luminance tercile, where local luminance is the mean Y over the box dilated ×1.5.
- Null scale: s_bin = 1.4826·MAD(ℓ_raw1 − ℓ_raw2), computed within the bin.
- Score:

  z_k(o) = (ℓ̄_raw(o) − ℓ_k(o)) / (s_bin·√(3/4))

  with ℓ̄_raw the mean of the two raw seeds. The √(3/4) factor makes var = 1 under H0: Var(ℓ̄_raw − ℓ_k) = 1.5σ², while Var(ℓ_raw1 − ℓ_raw2) = 2σ².
- Calibration check: a third raw seed, used as a fake enhancer, must give z ≈ N(0,1), i.e. P(z > 2) ≈ 2.3%.
- Gate: g_k(o) = σ((z_k − τ)/T_g), with τ = 2 and T_g = 0.5. If several k pass, renormalise among them with softmax(z_k).
- Teacher weights: w_raw(o) = max(0, 1 − Σ_k g_k(o)) and w_k(o) = g_k(o). Raw is the default.

**Step 8. C3 utility head (training only).**
- RoIAlign 7×7 on the student's neck P3–P5 for each GT box, then a 2-layer MLP (256 hidden), predicting ẑ ∈ ℝ^K.
- Loss: Huber(ẑ, z), δ = 1, weight λ_u = 1.
- Gradients are stopped into the detector by default (ablate).
- On unlabeled images:
  - Pseudo-boxes come from the full-data raw YOLOv8m teachers, fused with WBF, conf ≥ 0.5 (**threshold is my choice**, sweep {0.3, 0.5, 0.7}).
  - Gate = σ((ẑ − τ)/T_g).
  - Apply only feature and cls KD, with no box/detection loss on pseudo-labels, so as not to contaminate supervised training.
  - Mix ratio 1:1 labelled:unlabeled per batch.
- The head is deleted before export, so zero inference cost.

### Gaps
- M-AID's and CA-MKD's exact weighting equations and temperatures were not re-read (arXiv and par.nsf.gov blocked). Re-read before writing Related Work and implementing the "in-sample M-AID-style gate" baseline.
- The LD temperature τ = 10 and the Retinexformer parameter count are from memory (**UNVERIFIED**).
- No public Ultralytics-native implementation of box-masked FGD/PKD for YOLOv8 was found. Community CWD/MGD patches exist but are version-specific.

---

## Q2. Teacher plan: which teachers, K-fold scheme, number of runs, GPU-hours

### Takeaway
Teachers:
- YOLOv8m raw ×2 seeds (plus a 3rd raw seed for null calibration only);
- YOLOv8m trained on Zero-DCE++, Retinexformer and **corrected** SCI outputs.

Use **K = 3 folds**. K = 5 makes each teacher train on 80% vs 67% of the data, but costs 5/3× more. Total teachers per dataset: 15 cross-fitted + 3 null-calibration + 3 reliability + 5 full-data. For ExDark this is ≈17 full-run equivalents, ≈50–60 GPU-h on a 24 GB GPU. Measure the real time in week 1.

### Cited Findings
- The original design was 3 folds × (2 raw seeds + 3 enhancers) = 15 cross-fitted YOLOv8m runs plus 5 full-data teachers. Over 10 weeks the budget was ≈550 GPU-hours on one 24 GB GPU. SCI was to be dropped until fixed. KinD++ and LIME were the original enhancer picks. — lit-05 §3/§5; [PUBLICATION_ROADMAP §1.2](/home/user/lowlight/research/PUBLICATION_ROADMAP.md)
- The earlier pipeline's SCI was wrong: it saved the illumination map `i`, not `r`. Correct SCI output has mean ≈ 143 and is ≥ input pixel-wise. — [PUBLICATION_ROADMAP §0](/home/user/lowlight/research/PUBLICATION_ROADMAP.md)
- The Zero-DCE++ official test script crops to multiples of 12 from the top-left, so boxes must be checked, or pad instead. — [PUBLICATION_ROADMAP §3](/home/user/lowlight/research/PUBLICATION_ROADMAP.md)
- Official enhancer repos are live: [Zero-DCE_extension](https://github.com/Li-Chongyi/Zero-DCE_extension) (HEAD 09f202b), [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) (HEAD 1e9a0ef), [SCI](https://github.com/vis-opt-group/SCI) (HEAD f6f88fd). — git ls-remote 2026-11-03
- Teacher capacity is the most reliable source of KD gain. A YOLOv8m teacher with an equally large raw-teacher control is required. The original YOLOv8n P3 teachers were all below raw, leaving little KD headroom. — lit-05 §2, §4
- Ultralytics recommends a teacher one or two sizes up, e.g. n←s and s←m. — [Ultralytics KD guide](https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/knowledge-distillation.md)

### Inferences
**Teacher set (per dataset)**

| ID | Model / input | Purpose | Runs |
|---|---|---|---|
| R1, R2 | YOLOv8m, raw, seeds 0/1 | Raw baseline for z; raw KD teacher | 3 folds ×2 + 2 full |
| R3 | YOLOv8m, raw, seed 2 | Null calibration ("fake enhancer") and raw-seed-ensemble-gate control | 3 folds + 1 full (the full model is needed for the ensemble-gate control) |
| E1 | YOLOv8m on Zero-DCE++ (curve-based, tiny) | Enhanced teacher | 3 folds + 1 full |
| E2 | YOLOv8m on Retinexformer (LOL-v1 weights; transformer, strongest PSNR family) | Enhanced teacher | 3 folds + 1 full |
| E3 | YOLOv8m on SCI-corrected `r` (self-calibrated, unsupervised) | Enhanced teacher | 3 folds + 1 full |
| E1′ | YOLOv8m Zero-DCE++, seed 1 | Test–retest reliability of z_k | 3 folds |

- Diversity rationale: one curve-estimation, one Retinex-transformer, one self-supervised illumination method. KinD++/LIME are optional swaps.
- Teachers are trained **on enhanced images** (P3 protocol), COCO-pretrained, with the same recipe as raw: 100 epochs, imgsz 640 (DarkFace 1024, see Q3).

**K choice**
- K = 3 by default.
- K = 5 only in an ablation, on ExDark with E1 only. It tests whether utility estimates depend on teacher training-set size (a reviewer question).
- K = 1 means in-sample = the M-AID-style gate baseline (Q4).

**Run count and GPU-hours (estimates; re-time in week 1)**
- One YOLOv8m run on ExDark: 5.9k images, 100 epochs, 640, AMP, RTX 4090/3090-class. I estimate ≈2.5–3.5 h.
- Fold runs train on 2/3 of the data, so ≈0.67× each.
- ExDark: (2+1+3+1 = 7 teacher types × 3 folds × 0.67) + 6 full ≈ 14 + 6 = **≈20 run-equivalents ≈ 50–70 GPU-h**.
- DarkFace at 1024: ≈1.8× per run, so ≈90–120 GPU-h. Option: drop E1′ and use R3 only on DarkFace.
- Third dataset (NOD/BDD-night subset): ≈50 GPU-h.
- Utility extraction (OOF inference) is <1 GPU-h per dataset.

### Gaps
- Real per-epoch timing of YOLOv8m on ExDark/DarkFace was not found in any source. It must be measured in week 1.
- The Retinexformer training-set choice for low-light *detection* images (LOL-v1 vs LOL-v2-real vs SDSD weights) has no published guidance. Pick by the P3 mAP of teachers on fold 0.

---

## Q3. Datasets and splits

### Takeaway
Use four datasets:
1. **ExDark, MAET 80/20 split** (5,896 / 1,467): the standard benchmark, comparable to AMIEOD/SGLDet/MAET.
2. **DarkFace**:
   - 6,000 labelled images, YOLA 5,400/600 split, comparable to YOLA/FRBNet;
   - 9,000 unlabeled images (10,400 in another version of the paper) for C3.
3. **NOD**: 7.2k images, 46.7k boxes, 3 classes, extreme low light, plus 0.9k unannotated Sony images.
4. **BDD100K-night**: a large night-time driving set and a second unlabeled pool. LLVIP-visible is an optional fourth.

Headline numbers go on DarkFace and NOD, where enhancement utility is large. ExDark is the standard comparison.

### Cited Findings
- **ExDark MAET split:** 5,896/1,467. YOLOv3-608 at 25 epochs gives 76.4 → SGLDet 78.6. AMIEOD's setting (YOLOv3-640, 100 epochs) gives 76.5 → 82.1, and **raw-input 81.9**. — lit-05 §4
- **ExDark YOLA split:** 3,000/1,800/2,563. YOLOv3 71.0 → YOLA 72.7 / FRBNet 74.9; TOOD 72.5 → YOLA 75.2 / FRBNet 75.4. — lit-05 §4. The YOLA README lists ExDark YOLOv3 72.7, TOOD 75.3; DarkFace YOLOv3 61.5, TOOD 67.5 (67.5→68.9 with a COCO-pretrained YOLA). — [YOLA README](https://github.com/MingboHong/YOLA)
- **DarkFace official description:** 6,000 labelled nighttime images for train/val, **9,000 unlabeled** images from the same setting, and a 4,000-image hold-out test set (labels not public). — [DARK FACE project page](https://flyywh.github.io/CVPRW2019LowLight); the UG2+ v2 paper says **10,400** unlabeled — [UG2+ arXiv 1904.04474](https://arxiv.org/pdf/1904.04474v2) (conflict; count the files you download).
- **DarkFace splits** in the literature:

  | Source | Split | Detector | Baseline → method |
  |---|---|---|---|
  | SGLDet | 5,500/500 | DSFD | 59.7 → 76.6 |
  | YOLA | 5,400/600 | YOLOv3 | 60.0 → 61.5 |
  | YOLA | 5,400/600 | TOOD | 62.1 → 67.4 |
  | FRBNet | 3:1:1 | YOLOv3 | 54.8 → 57.7 |
  | FRBNet | 3:1:1 | TOOD | 57.0 → 65.1 |
  | AMIEOD | — | YOLOv3 | 63.0 → 64.3 |

  — lit-05 §1, §4
- **NOD (BMVC 2021):** "more than 7K images and 46K annotated objects", classes person/bicycle/car. Sony set: 3.2k annotated + 0.9k unannotated images. Nikon set: 4.0k annotated images. Download requires a Google form. — [NOD repo](https://github.com/igor-morawski/NOD)
- **LLVIP:** AMIEOD reports 91.4 → 92.8. — lit-05 §1; [LLVIP repo](https://github.com/bupt-ai-cz/LLVIP) (live, HEAD c1a655c)
- **nuImages-night:** YOLOv8-m 52.4 → SGLDet 55.3; SGLDet split 3,591/894. — lit-05 §1, §5
- The first-place UG2+ 2021 solution treated DarkFace as "(semi-)supervised" with the unlabeled set, so prior semi-supervised use exists in a challenge context. — [ar5iv 2107.00818](https://ar5iv.arxiv.org/html/2107.00818)

### Inferences
| Dataset | Split for Paper 2 | Role | imgsz | Notes |
|---|---|---|---|---|
| ExDark (12 classes) | MAET 80/20 (5,896/1,467) via [ICCV_MAET](https://github.com/cuiziteng/ICCV_MAET); YOLA split as a secondary row | Standard benchmark; all controls | 640 | Report by ExDark's 10 lighting conditions in the supplement |
| DarkFace (face) | YOLA 5,400/600 (split files in YOLA's Drive package); unlabeled pool for C3 | Headline + C3 | 1024 (small faces; ablate 640) | Never use the 4k hold-out (no labels) |
| NOD (3 classes) | **UNVERIFIED** whether an official train/test split exists. If not, a fixed 80/20 stratified by camera; release the split file | Headline (extreme dark) + 0.9k unlabeled | 640 | Raw RGB is JPEG-processed; check the format |
| BDD100K-night | Night subset by `timeofday=night`; downsample to ≈10k train to fit compute; official val-night for test | Generality + large unlabeled pool (train-night unlabeled) | 640 | Night-image counts **UNVERIFIED**; compute from labels |
| LLVIP-visible | Official train/test, visible only | Optional fourth (nice-to-have) | 640 | Single class, near-saturated (>90 mAP50), low headroom |

Data hygiene:
- For cross-fitting, folds are drawn from the **training** split only. Stratify by image-level darkness tercile and by class presence.
- Fix seeds; save `folds.json`.

### Gaps
- An official NOD train/test split and BDD100K night-image counts could not be verified in this session.
- The DarkFace unlabeled count conflicts between sources (9,000 vs 10,400).

---

## Q4. Baselines: code links, how to reproduce, must-have vs nice-to-have

### Takeaway
Must-have: every zero-inference-cost competitor run **in our own YOLOv8n pipeline under identical epochs**:
- raw;
- photometric augmentation;
- P4 mixed training;
- AMIEOD-style multi-view (raw inference);
- SGLDet reimplementation (code still unreleased);
- KD controls: single raw-teacher KD, ungated multi-teacher, random gate, in-sample M-AID gate, raw-seed-ensemble gate.

Plug-in modules (YOLA, FeatEnHancer, PE-YOLO, FRBNet) are cited at their own protocols. YOLA is also ported to YOLOv8n, since it is a small module. DAI-Net and IA-YOLO are cited only.

### Cited Findings
- **AMIEOD** code is public: [jayfantasyxc/AMIEOD](https://github.com/jayfantasyxc/AMIEOD) (HEAD 120c6c9).
  - It is a **YOLOv5-style** codebase (`train.py`, `val.py`, `models/`, `utils/`) with `train_AMIEOD_stage1.py`, `train_stage2.py` and `val_AMIEOD.py`.
  - README commands: stage 1 at 100 epochs, batch 8, `--cfg yolov3.yaml`; stage 2 at 30 epochs, batch 1; data prepared "as guided by MAET".
  - Weights are on Baidu Pan (password `vsst`).
  - The README calls `train_stage1.py`, but the file is `train_AMIEOD_stage1.py`.

  — [AMIEOD repo](https://github.com/jayfantasyxc/AMIEOD)
- **SGLDet** (ICLR 2026) [gw-shin/SGLDet](https://github.com/gw-shin/SGLDet) contains only README/assets/LICENSE. Checklist: "[ ] Code coming soon", "[ ] Release pre-trained model weights" (last commit 2026-04-25). Its method is a training-only enhancing + denoising + Fourier fusion pipeline, "completely removed at inference time". — [SGLDet README](https://github.com/gw-shin/SGLDet). Loss: MSE with λ = 0.01; it collapses at λ ≥ 0.05. — lit-05
- **YOLA** (NeurIPS 2024): [MingboHong/YOLA](https://github.com/MingboHong/YOLA), MMDetection-based. It provides dataset classes `exdark_voc.py` / `dark_face.py` and released checkpoints (Google Drive). — [YOLA README](https://github.com/MingboHong/YOLA)
- **FRBNet** (NeurIPS 2025): [Sing-Forevet/FRBNet](https://github.com/Sing-Forevet/FRBNet) (HEAD 5e4eda3), MMDetection, YOLOv3 on ExDark. The authors list a non-MMDet version as not yet released. — [FRBNet search summary](https://neurips.cc/virtual/2025/poster/119034); git ls-remote
- **FeatEnHancer** (ICCV 2023): [khurramHashmi/FeatEnHancer](https://github.com/khurramHashmi/FeatEnHancer) (HEAD 33da743). DarkFace RetinaNet 47.3→47.2 and FQ-RCNN 67.5→69.0. — lit-05 §4
- **PE-YOLO** (ICANN 2023): [XiangchenYin/PE-YOLO](https://github.com/XiangchenYin/PE-YOLO) (HEAD 9fcc48d). YOLOv3 + Laplacian-pyramid enhancement; ExDark ≈78.0 mAP, 53.6 FPS (reported). — [PE-YOLO paper page](https://deepai.org/publication/pe-yolo-pyramid-enhancement-network-for-dark-object-detection)
- **IA-YOLO**: [wenyyu/Image-Adaptive-YOLO](https://github.com/wenyyu/Image-Adaptive-YOLO) (live). **DAI-Net**: [ZPDu/DAI-Net](https://github.com/ZPDu/DAI-Net) (live). **MAET**: [cuiziteng/ICCV_MAET](https://github.com/cuiziteng/ICCV_MAET) (live). DAI-Net on ExDark MAET YOLOv3-608: 78.3; MAET 77.7. — lit-05 §4
- **YOLA-Naive:** ungated consistency can hurt, e.g. −0.9 on ExDark TOOD and −0.8 on DarkFace YOLOv3. — lit-05 §1
- **AMIEOD mixed-view training without DGRL:** 76.9–78.3 on raw input; with DGRL, raw input reaches 81.9. — lit-05 §1

### Inferences
| # | Baseline | Must / Nice | How to reproduce | Inference cost |
|---|---|---|---|---|
| B1 | Raw YOLOv8n | Must | Ultralytics defaults, 100 epochs | 0 |
| B2 | Raw + extra epochs (same total updates as the KD runs, e.g. 130 epochs) | Must | `epochs=` | 0 |
| B3 | Photometric augmentation (`hsv_v` 0.4→0.7, random gamma ∈ [0.4, 2.5], Poisson-Gaussian noise) | Must | Custom albumentations transform | 0 |
| B4 | P4 mixed training (each image randomly raw or one of the 3 enhancers, p = 0.5) | Must | Online enhancer in the dataloader | 0 |
| B5 | AMIEOD-style multi-view + per-image best-view regression (DGRL), raw inference | Must | Port DGRL to YOLOv8n; also run the **official** repo (YOLOv3) on MAET to anchor 81.9 | 0 |
| B6 | SGLDet reimplementation (training-only decoder, λ = 0.01, SCI-phase + denoised amplitude target) | Must | Reimplement on the YOLOv8 neck; state "our reimplementation" | 0 |
| B7 | Single raw-teacher KD (R1, YOLOv8m), same C2 losses | Must | Our code, gate ≡ raw | 0 |
| B8 | Ungated multi-teacher KD (uniform average of raw + 3 enhanced) | Must | Our code, w_t = 1/4 | 0 |
| B9 | Random gate, matched on-rate per (size × lum) bin | Must | Shuffle z within bins | 0 |
| B10 | In-sample M-AID/CA-MKD-style gate (full-data teachers' own GT losses, softmax over −ℓ, no null) | Must | Same code, K = 1, z from in-sample ℓ | 0 |
| B11 | Raw-seed-ensemble gate (R1/R2/R3 as "enhancers" through the identical gate) | Must | Same code | 0 |
| B12 | Native Ultralytics KD (score-weighted L2, `dis` = 6) from R1 | Must (cheap, shows our C2 > stock) | `distill_model=R1.pt` | 0 |
| B13 | YOLA module ported to YOLOv8n | Nice→Must if reviewers target ICCV | Insert the YOLA IIM before the backbone; cite their mmdet numbers separately | Small (+module) |
| B14 | Enhance-then-detect (each enhancer + YOLOv8n trained on enhanced data, P3) | Must (already trained as teachers' small siblings) | — | +enhancer |
| B15 | FeatEnHancer, PE-YOLO, FRBNet, IA-YOLO, DAI-Net, MAET, TorchAdapt | Nice: cite published numbers in a separate block "different detector/protocol", never mixed with our rows | — | — |

### Gaps
- No public SGLDet code as of the 2026-04-25 commit, so a faithful reproduction is uncertain. Label it "reimpl." and report a sanity check: YOLOv3 on MAET should land near 78.6.
- Whether YOLA's split files are inside its Drive package was not verified (Drive not accessible).

---

## Q5. Ablations, sensitivity, efficiency and statistical reporting

### Takeaway
Ablations must isolate each claimed ingredient:
- cross-fitting;
- null normalisation;
- gate vs ensemble;
- C2 loss form;
- C3.

Run them on ExDark with 1 seed (3 seeds for the key rows) plus DarkFace for C3. Report mean ± std over 3 seeds and paired-bootstrap 95% CIs over test images, Holm-corrected across baselines. The efficiency table must show *training* cost as well as params, FLOPs and FPS.

### Cited Findings
- The required statistical protocol is 3 seeds and paired-bootstrap 95% CIs on the 1,467-image MAET test set, under the same epochs and teacher size. — lit-05 §4; [PUBLICATION_ROADMAP §1.2](/home/user/lowlight/research/PUBLICATION_ROADMAP.md)
- What reliably adds mAP, ranked: teacher capacity; masked FPN feature KD (FGD/PKD) over logit-only, plus DFL/box KD; all FPN levels, with P3 carrying small-object gains; multi-view training as the base; 2–3 diverse teachers; photometric augmentation. — lit-05 §4
- Sunshine-to-Rainstorm: per-instance weighting added +2.5 over unweighted KD, in 3D detection. — lit-05 §1
- Theory predictions to test:
  - P1: the gain shrinks with data size;
  - P2: the gain concentrates on dark, small objects;
  - P3: a raw-seed ensemble explains part of the gain.

  — [PUBLICATION_ROADMAP §1.2](/home/user/lowlight/research/PUBLICATION_ROADMAP.md)
- YOLOv8n/s/m params and GFLOPs (8.9/28.8/79.3 at 640) are in the model YAML. — [yolov8.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml)

### Inferences
**Ablation table A (ExDark MAET, YOLOv8n; 3 seeds on rows marked \*)**

| Factor | Settings |
|---|---|
| Gate type\* | none (B8) / random (B9) / in-sample (B10) / raw-seed-ensemble (B11) / **cross-fitted + null (ours)** / oracle (in-sample on test, upper bound, supplement only) |
| Cross-fitting\* | off (K = 1) / K = 3 / K = 5 (E1 only) |
| Null normalisation\* | none (raw Δℓ) / global scale / per-bin MAD (ours) / per-bin + seed-pair √(3/4) correction |
| Threshold τ | 0, 1, 1.5, **2**, 3; plus T_g ∈ {0.25, **0.5**, 1, hard}; plus an FDR-controlled hard gate (Benjamini–Hochberg q = 0.1) |
| Teachers: number/type | raw only; +1 enhancer (each of E1/E2/E3 alone); +2; +3 (ours); teacher without corrected SCI vs with |
| Teacher size | v8s / **v8m** / v8l (v8l only for raw + best enhancer) |
| Student size | v8n (main) / v8s |
| KD target teachers | full-data teachers (default) vs OOF teachers as targets |
| C2 loss form | MSE vs PKD-standardised (ours) vs CWD (τ = 1) vs native score-weighted L2; feature-only / logit-only / both; ± DFL KD; ± background term |
| FPN levels | P3 only / P4 / P5 / P3+P4 / all (ours); size-assigned vs all-levels masks |
| Loss weights | λ_fg ∈ {1, 3, 6, 10}; λ_cls ∈ {0.5, 1, 2}; τ_LD ∈ {1, 5, 10} |
| Base recipe | KD on plain training vs on the AMIEOD-style multi-view base |
| Augmentation order | online enhancement after augmentation (default) vs offline enhance-first |

**Ablation table B (DarkFace; C3)**
- C3 off / on (labelled only) / on + unlabeled.
- Unlabeled amount: 0, 25, 50, 100% of the pool; also BDD-night unlabeled as cross-domain.
- Pseudo-box conf: 0.3/0.5/0.7.
- Stop-grad on/off.
- Utility-head quality metrics:
  - Spearman ρ(ẑ, z) on held-out fold objects;
  - AUROC for predicting z > τ;
  - "recovery ratio" = (AP_label-free − AP_ungated) / (AP_GT-gate − AP_ungated); GNG-3 target ≥ 0.7.

**Sensitivity / analysis experiments**
- Data-scaling curve (prediction P1): 10/25/50/100% of ExDark training data, for raw / ungated KD / ours. 3 seeds at 25%/100%, 1 seed otherwise.
- Gain by object size (COCO S/M/L areas) and by local-luminance tercile (prediction P2).
- Ensemble-vs-enhancement decomposition (prediction P3): Δ_total = (ours − no-KD) = (B11 − no-KD) [ensembling/capacity] + (ours − B11) [enhancement-specific].
- z reliability: Spearman of z_E1 vs z_E1′ (two enhancer seeds), and the null false-positive rate using R3.
- TIDE error breakdown (cls/loc/bkg/miss) for raw vs ours, in the supplement.

**Efficiency table**
- Columns: params, GFLOPs@640, FPS (batch 1, FP16, same GPU, PyTorch and TensorRT), latency ms, **training GPU-h including teachers**, extra training memory.
- Rows: YOLOv8n raw / ours (identical inference) / enhancer + YOLOv8n (add enhancer FLOPs) / YOLA-v8n / FeatEnHancer / PE-YOLO (reported) / AMIEOD with selector (ResNet-50 adds cost) vs raw-input.

**Statistics**
- 3 seeds (0, 1, 2) for every main-table row; report mean ± std of mAP50 and mAP50-95.
- Paired bootstrap: resample test **images** with replacement (B = 2,000; 10,000 for headline comparisons). Recompute mAP for both methods on the same resample, pooling the seed-averaged predictions, or bootstrap per seed and then average. Report the 95% percentile CI of Δ.
- Holm correction across the ~12 baseline comparisons per dataset.
- Use [faster-coco-eval](https://github.com/MiXaiLL76/faster_coco_eval) or pycocotools for speed. (The faster-coco-eval link was not verified in this session: **UNVERIFIED** URL.)
- Report the number of runs and the selection protocol: hyper-parameters tuned on **fold-0 validation within the training set**, never on test.

### Gaps
- No source fixes the right bootstrap B or a seed-pooling convention for detection. The above is a defensible choice, not a published standard.

---

## Q6. Figures, tables, section outline, page limits, supplementary material, reproducibility checklist

### Takeaway
Plan for 8 pages + refs:
- ICCV 2027: 8 pages excluding references, per the ICCV 2025 rule; the 2027 call is not out.
- ACM MM 2027: 8 pages + ≤2 reference-only pages, per the ACM MM 2026 rule.
- AAAI-28: 7 pages + references + checklist, per the AAAI-26 rule.

Write to 7 pages of core content so the AAAI version needs no restructuring. Six figures and five tables fit.

### Cited Findings
- **ICCV 2025:** papers limited to eight pages including figures and tables; extra pages may contain only references. No ICCV 2027 call or author kit was found. An aggregator projects the **ICCV 2027 deadline as ~8 March 2027** (abstract ~1 March), Hong Kong, 2–8 Oct 2027. This is **UNVERIFIED/projected**. — [ICCV 2025 Author Guidelines](https://iccv.thecvf.com/Conferences/2025/AuthorGuidelines); [PaperPilot ICCV 2027](https://www.getpaperpilot.com/deadlines/iccv-2027.html); [OpenCurious ICCV 2027](https://www.opencurious.com/ai-conference-deadlines/iccv-2027)
- **ACM MM 2026:** main track 8 pages of content + up to 2 pages for references only; overlength papers are desk-rejected. — [ACM MM 2026 CFP](https://2026.acmmm.org/site/cfp-guidelines.html); [ACM MM 2026 author instructions](https://2026.acmmm.org/site/author-instructions.html). The ACM MM 2027 deadline (~1 Apr 2027) is **UNVERIFIED** (roadmap estimate).
- **AAAI-26:** up to 7 pages of technical content plus extra pages for references and the reproducibility checklist. **AAAI-27** requires all authors to complete a reproducibility checklist. AAAI-27 main-track papers were due ~28 Jul 2026 (third-party template site), so **AAAI-28 ≈ late Jul 2027 (UNVERIFIED)**. — [AAAI-26 submission instructions](https://aaai.org/conference/aaai/aaai-26/submission-instructions/); [AAAI-27 main track call](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/); [AAAI-27 template page](https://www.thesify.ai/templates/aaai-2027-latex-template)
- AAAI reproducibility checklist (historical version) — [AAAI-23 reproducibility checklist](https://aaai-23.aaai.org/reproducibility-checklist/)

### Inferences
**Section outline (8-page budget; AAAI cut in brackets)**

| § | Content | Pages |
|---|---|---|
| 1 Intro | Problem; "computable privilege" insight (enhancer = deterministic f(x): bias, not information, I(Y; f(X)) ≤ I(Y; X)); utility is small, noisy and object-level, so it must be estimated OOF against a seed null; contributions C1–C3 + falsifiable predictions; Fig. 1 | 1.0 |
| 2 Related | LLIE-for-detection (AMIEOD, SGLDet, YOLA, FRBNet, TorchAdapt); detection KD (FGD, PKD, CWD, MGD, LD); multi-teacher instance weighting (M-AID, CA-MKD, AMTML, Sunshine-to-Rainstorm, PATT); privileged-information/LUPI; counterfactual restoration utility (TaskGuard) | 0.75 [0.6] |
| 3 Method | 3.1 Setup and notation; 3.2 C1 cross-fitted, null-calibrated utility (equations for ℓ, z, the √(3/4) correction, gate); 3.3 C2 standardised box-masked + logit KD; 3.4 C3 amortised utility head + semi-supervised gating; 3.5 Predictions P1–P3; Fig. 2 | 2.25 [2.0] |
| 4 Experiments | 4.1 Datasets/protocol/baselines; 4.2 Main results (Tables 1–2); 4.3 Gate analysis, ensemble decomposition (Table 3, Fig. 3–4); 4.4 Ablations (Table 4); 4.5 C3 + unlabeled (Fig. 5); 4.6 Generality (RT-DETR, v8s) + efficiency (Table 5); 4.7 Qualitative (Fig. 6) | 3.5 [3.1] |
| 5 Limitations + Conclusion | Training cost; dependence on teacher quality; null assumes seed noise ≈ teacher-swap noise | 0.5 [0.3] |

**Figures**
1. **Fig. 1 teaser** (one column).
   - Left: one ExDark image with two objects, one where the enhanced teacher is significantly better (green, z = 3.4) and one where it is worse or insignificant (grey, z = 0.3).
   - Right: a bar chart of mAP50 for raw / ungated KD / ours at **identical inference cost** (YOLOv8n, FPS label).
2. **Fig. 2 architecture** (full width). Three panels:
   - (a) **Offline utility estimation:** training set → K folds. For each fold f, teachers {R1, R2, E1–E3} train on the other folds and infer on fold f. Per-object losses ℓ go to "null calibration" (R1−R2 disagreement per size × luminance bin, MAD), which outputs z_k(o).
   - (b) **Training:** a raw image enters the student (YOLOv8n). In parallel, frozen full-data teachers receive raw / E_k(x) (online enhancer icons). Their neck features P3–P5 feed a "box mask × gate g_k(o)" block, then the PKD-standardised feature loss and cls/DFL logit loss, then into the student. The background term comes from the raw teacher only. A utility head on student RoI features is supervised by z on labelled data and produces ẑ, which gates pseudo-boxes from unlabeled images.
   - (c) **Inference:** raw → student → boxes ("teachers, enhancers, utility head removed: +0 FLOPs").
   - Arrow colours: data (black), gradient (red dashed), no-grad (grey).
3. **Fig. 3 utility distributions:**
   - (a) Histogram of z_k for each enhancer vs the R3 null, log-y, with the τ = 2 line.
   - (b) Share of z > 2 vs object size (log area) and local luminance, as 2-D heat maps or binned lines with bootstrap CIs.
4. **Fig. 4 gate visualisation:** 4–6 images with boxes coloured by argmax teacher (raw / ZDCE / Retinexformer / SCI) and opacity ∝ g. Include enhanced-view thumbnails.
5. **Fig. 5 scaling curves:**
   - (a) Data-scaling: Δ mAP50 over raw vs training fraction (10–100%) for ungated KD and ours (tests P1).
   - (b) Unlabeled scaling: DarkFace mAP vs unlabeled amount (0–100%).
   - (c) Training curves (val mAP vs epoch) for raw / ungated / ours, in the supplement if space is short.
6. **Fig. 6 qualitative detections:** raw input (dark), with ours vs ungated KD vs AMIEOD-style vs raw. Show misses and false positives on DarkFace tiny faces and NOD; 3 datasets × 2 images.
7. Per-size AP bars (S/M/L × methods), either in Fig. 3 or the supplement.

**Tables**
- **T1 Main ExDark (MAET):** all must-have baselines in our pipeline (mAP50, mAP50-95, APs, Δ vs raw with CI), plus a separate block of published numbers (YOLOv3 protocols) clearly labelled.
- **T2 DarkFace + NOD (+ BDD-night):** same columns; C3 variants.
- **T3 Gate analysis:** gate types × {mAP50, gate on-rate, null FPR, reliability ρ}, plus the decomposition row (ensemble vs enhancement-specific).
- **T4 Ablations:** compact multi-block table (C1 components, C2 loss form, levels, teachers, τ).
- **T5 Generality and efficiency:** YOLOv8s student, RT-DETR student; params/FLOPs/FPS/training GPU-h.

**Supplementary material**
- Full derivation of the null scaling;
- per-class and per-lighting-condition (ExDark 10 conditions) results;
- YOLA-split ExDark results;
- K = 5 and teacher-size ablations;
- TIDE;
- hyper-parameter sweeps;
- all seeds' raw numbers;
- bootstrap code;
- teacher P3 mAPs;
- failure cases where the gate is wrong;
- fold files;
- compute statement;
- the enhancer implementation notes (SCI `r` fix, Zero-DCE++ padding).

**Reproducibility checklist items** (AAAI-style; also good practice for ICCV/MM)
- Code + configs + `folds.json` + per-object z tables released, AGPL-3.0 because Ultralytics-derived.
- Exact dataset versions and splits.
- Seeds.
- Hyper-parameter search ranges and selection on train-internal validation.
- Number of runs and mean ± std.
- Significance-test description.
- Compute (GPU type, GPU-h incl. teachers).
- Pretrained weights' sources (COCO YOLOv8, enhancer weights).
- Licences of datasets: ExDark is BSD-3 per its repo (**UNVERIFIED**); NOD is form-gated; DarkFace is research-only (**UNVERIFIED**).

### Gaps
- No official ICCV 2027, ACM MM 2027 or AAAI-28 calls exist yet (as of 2026-11-03). All 2027 page limits and deadlines are projected from the 2025/2026 rules.
- Not found: CVF supplementary-material size limits, and whether ACM MM requires a formal reproducibility checklist.

---

## Q7. Week-by-week schedule (12 weeks) with go/no-go checkpoints and expected numbers

### Takeaway
The plan runs 12 weeks from 9 Nov 2026 to 31 Jan 2027, leaving February for writing and ICCV (~8 Mar 2027, projected). Three hard gates:
- **GNG-1 (end of week 3):** utility is not noise.
- **GNG-2 (end of week 4):** gate > ungated by ≥ 0.8 and > raw-seed-ensemble gate by ≥ 0.5 on ExDark, confirmed with 3 seeds.
- **GNG-3 (end of week 7):** C3 recovers ≥ 70% of the GT-gate gain, and unlabeled data adds ≥ +1.0 on DarkFace.

Expected numbers are ranges derived from published baselines, not measurements.

### Cited Findings
- Original gates:
  - GNG-1 (week 2): z > 2 must be more common than under the raw-vs-raw null.
  - GNG-2 (week 3): ≥ +0.8 over ungated **and** ≥ +0.5 over the raw-seed-ensemble gate on ExDark.
  - GNG-3 (week 5): ≥ 70% recovery and ≥ +1.0 from unlabeled data on DarkFace.

  ICCV only if every gate passes with ≥ +1.5 over the best control on 3 datasets; otherwise ACM MM 2027 / AAAI-28. Fallback if GNG-2 fails: reframe as "enhancement as computable privileged information" if ungated enhanced-teacher KD beats multi-view by ≥ +1; else fold into Paper 1. — [PUBLICATION_ROADMAP §1.2](/home/user/lowlight/research/PUBLICATION_ROADMAP.md); lit-05 §5
- Realistic targets for YOLOv8n:
  - +2–3 mAP50 over no-KD raw, mostly from the larger teacher;
  - +0.8–1.5 (ExDark) and +1.5–3 (DarkFace) over the strongest zero-cost control;
  - gate vs ungated +0.3–1.0, "the main risk".

  — lit-05 §4
- AMIEOD-style training raised YOLOv3 by up to +5.4 on raw input (76.5 → 81.9). — lit-05 §1, §4

### Inferences
| Week (start) | Work | Checkpoint and expected numbers |
|---|---|---|
| **W1** (9 Nov) | Fix SCI (`r`), Zero-DCE++ padding. MAET split + folds.json (ExDark). DarkFace 5,400/600 + unlabeled pool; NOD download (form). Time one YOLOv8m epoch. Train YOLOv8n raw ×3 seeds, YOLOv8m raw full-data R1. Stand up `GatedDistillationModel` skeleton; unit tests (hooks fire, `fuse()` param count = 3,157,200). Run B12 (native KD) as a smoke test. | YOLOv8n raw MAET mAP50: expected ≈70–76 (**no published v8n MAET number found**). R1 (v8m) should exceed v8n by ≥ +3; if not, KD headroom is too small, so try v8l. Native KD B12 ≥ raw. |
| **W2** (16 Nov) | Cross-fitted teachers ExDark: R1, R2, R3 × 3 folds; E1 × 3 folds (≈12 runs). Per-object ℓ extraction; choose ℓ definition (a) vs (b) by seed test–retest ρ. | Raw-seed ℓ test–retest Spearman ρ ≥ 0.5 expected (otherwise per-object losses are too noisy at all: switch to (b), or average 3 raw seeds). |
| **W3** (23 Nov) | E2, E3, E1′ × 3 folds; full-data E1–E3. Compute z; Fig. 3 draft. **GNG-1.** | **GNG-1 pass:** for ≥ 1 enhancer, P(z > 2) ≥ 2× the R3-null rate (null ≈ 2–4%), CI excluding equality (bootstrap over images), **and** z reliability ρ(E1 vs E1′) ≥ 0.3, **and** z > 2 objects enriched among small/dark bins. **Fail:** stop Paper 2's method; write up the null in Paper 1. |
| **W4** (30 Nov) | Full ExDark control grid, 1 seed (B1–B12 + ours); then 2 more seeds for B8, B11, ours. AMIEOD-style multi-view base on v8n. **GNG-2.** | Expected: ungated KD (B8) ≈ raw + 1.5–2.5; ours ≈ B8 + 0.3–1.0. **GNG-2 pass:** ours − B8 ≥ +0.8 and ours − B11 ≥ +0.5, 3-seed mean, paired CI excluding 0. **Fail branch:** if B8 − B5 ≥ +1, reframe (AAAI/TCSVT); else fold into Paper 1. |
| **W5** (7 Dec) | DarkFace teachers (cross-fit, 1024 px): R1–R3, E1–E3 × 3 folds + full. Start NOD teachers if time. | DarkFace YOLOv8n raw mAP50 expected ≈55–65 (cf. YOLOv3 60.0 on the YOLA split). |
| **W6** (14 Dec) | DarkFace grid (B1, B4, B5, B7, B8, B11, ours) × 1 seed; ablations on ExDark (C2 loss form, levels, τ, K = 1/3/5). | Ours − strongest zero-cost control on DarkFace ≥ +1.5 expected if hypothesis P2 holds (small faces). |
| **W7** (21 Dec) | C3: utility head + semi-supervised KD on the DarkFace unlabeled pool. **GNG-3.** | ρ(ẑ, z) ≥ 0.4; recovery ≥ 70%; unlabeled adds ≥ +1.0 mAP50 on DarkFace. **Fail:** drop C3 → target ACM MM with C1 + C2. |
| **W8** (28 Dec, light week) | NOD (or BDD-night) teachers + grid (reduced: B1, B5, B8, B11, ours); BDD-night unlabeled for cross-domain C3. | NOD gains expected ≥ DarkFace-like (extreme dark). |
| **W9** (4 Jan) | Second student/detector: YOLOv8s student; RT-DETR-l student with RT-DETR-x teachers via custom encoder-feature hooks (feature-KD only; Ultralytics native KD does not support RT-DETR). SGLDet reimplementation; official AMIEOD run on MAET. | AMIEOD official YOLOv3 should reproduce ≈81.9 raw-input (±1); otherwise document. |
| **W10** (11 Jan) | 3 seeds for all main-table rows on 3 datasets; data-scaling curve (10/25/50/100%). | **P1 check:** the gain at 25% > the gain at 100%. If not, drop the theory claim. |
| **W11** (18 Jan) | Analyses: gate maps, per-size/luminance AP, ensemble-vs-enhancement decomposition, TIDE, efficiency table, bootstrap + Holm. Re-search arXiv for new competitors since Nov 2026. | Venue decision: ICCV only if ≥ +1.5 over the best control on 3 datasets with CIs and all GNGs passed; else ACM MM (~1 Apr) / AAAI-28 (~late Jul). |
| **W12** (25 Jan) | Figures and tables final; draft §3–4; supplementary; code clean-up and release prep. | Full draft by 31 Jan; February for writing and polishing. |

**GPU-hour budget (estimate; one 24 GB GPU)**

| Item | GPU-h |
|---|---|
| Teachers: ExDark | ~60 |
| Teachers: DarkFace | ~110 |
| Teachers: NOD/BDD | ~50 |
| ExDark grid (≈36 student runs at ~2–3 h; KD runs dominated by teacher forwards: four YOLOv8m forwards ≈ 4 × 79 GFLOPs vs 3 × 8.9 for the student step) | ~90 |
| DarkFace grid | ~100 |
| Third-dataset grid | ~50 |
| Ablations | ~75 |
| C3 | ~40 |
| RT-DETR/v8s | ~55 |
| Scaling | ~25 |
| **Total** | **≈650 GPU-h** |

Over 12 weeks this is ≈35% utilisation of one GPU. That is feasible, but queue runs overnight and on weekends. Ways to save:
- skip enhanced-teacher forwards when no object in the batch is gated on (est. −30–50% KD cost);
- train fold teachers for 60 epochs (validate on fold 0 that ranking-relevant ℓ is stable).

### Gaps
- No published YOLOv8n mAP50 on the ExDark MAET split or the YOLA DarkFace split was found. Week-1 expected values are extrapolations from YOLOv3 numbers.
- Real training throughput on the user's GPU is unknown. All GPU-hour figures are estimates to be replaced after week 1.
- No check was run in this session for post-Nov-2026 competitor papers on per-object gated distillation from enhanced teachers; arXiv was unreachable. Re-run the novelty search in week 11.
