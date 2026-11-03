# Publication Roadmap v2: verified against full papers (2026-11-03)

v1 checked novelty only through search snippets, because arXiv was blocked. **v2 re-checks every idea against the full papers** (`literature/05–07`), confirms the SCI bug, and gives only the claims that survived.

> **No one can guarantee better results before the experiments are run.** What this plan does instead:
> 1. Builds every method on the **strongest recipe already published**.
> 2. Evaluates where enhancement utility is **largest**: darker datasets.
> 3. Puts **go/no-go checkpoints** early, so a weak idea is dropped in week 2–3, not month 6.

---

## 0. What changed from v1 (corrections)

| Earlier claim | Full-text verdict | Consequence |
|---|---|---|
| Paper 2's per-object, per-teacher, ground-truth-loss gate is new | ❌ **M-AID (IEEE T-IV 2022)** already weights distillation per instance, per scale and per teacher from each teacher's ground-truth loss. CA-MKD does the same for classification. | Paper 2 redesigned (§2): novelty moves to *how utility is estimated*. |
| Paper 1 headline: "the sign flips between P0 and P1/P2" | ⚠️ **LIME-Eval and the CVPRW 2026 LoViF paper** both ran frozen COCO detectors (P0) on ExDark: enhancement mostly did **not** help. On **DarkFace** it **did** help, for all 14 enhancers. | Headline becomes **protocol × darkness interaction**. |
| "P1−P3 = irreversible information loss" (data-processing inequality) | ⚠️ The DPI bounds Shannon information, not *usable* information. LIME-Eval shows **augmentation order** (enhance-first vs augment-first) moves enhanced-trained results back to raw level. | Rename to **residual (non-adaptable) penalty**, cite V-information, and control augmentation order. |
| Paper 3's "enhancement helps recognition but hurts localization" is new | ❌ **Morawski et al., IEEE TMM 2025:** on ExDark, ~16 enhancers lower YOLO-World mAP but raise CLIP accuracy. **SIC-CADS (AAAI 2024)** and **arXiv 2509.19875** already rescore detections with an image-level or VLM prior. | Paper 3 reframed as a **measurement paper**, conditional on a pilot. |
| A separate "no-regret enhancement" paper ≈ 5/10 | ❌ 3/10. TaskGuard's counterfactual harm map, Sun et al. 2022's detection-loss constraint and DBDiff's fusion map already exist. | **Not a standalone paper.** Its "regret rate" metric goes into Paper 1 and its harm signal into Paper 2. |
| DPI-Det | ❌ 3/10 overall; no component is new (`literature/02`) | Replaced by Paper 2. |

**Confirmed bug, SCI.**
- I re-ran SCI myself (own reimplementation, official weights) on the doc's own raw sample. The doc's "SCI" image matches SCI's **illumination map `i`** (medium weights), not the enhanced output `r`:

  | Statistic | Doc "SCI" | SCI-medium `i` |
  |---|---|---|
  | Mean | 63.8 | 64.4 |
  | Contrast (std) | 12.8 | 12.8 |
  | Intensity slope | 0.57 | 0.57 |
  | Intensity offset | 40.6 | 41.2 |
  | Mean absolute pixel difference | 1.1 / 255 | |

- The correct SCI output has mean ≈ 143 and is always ≥ the input pixel-wise.
- **Every SCI number in the doc (−0.149 mAP) must be redone by saving `r`.**
- The script is `analysis/sci_bug/`.

---

## 1. Final portfolio: 3 papers, each with a verified-novel core

| # | Paper | Venue | Novelty (full-text) | Confidence |
|---|---|---|---|---|
| **1** | *When Does Low-Light Enhancement Help Detection? A Protocol × Darkness Study with Controlled Interventions* | **Q1 journal:** Pattern Recognition, IEEE TCSVT, or IEEE TMM (where Morawski published) | 6–7/10 for a journal: four sub-claims are verified OPEN | **High** |
| **2** | *Distilling Computable Privilege: Counterfactual Enhancement Utility for Raw-Input Low-Light Detection* | **A\*:** ACM MM 2027 (~Apr 1) → AAAI-28 (~late Jul); ICCV 2027 (~early Mar) only if every checkpoint passes by February. Fallback: TCSVT / TIP. | 5/10, or 6/10 if the label-free / unlabeled part works | **Medium**; the go/no-go gates decide |
| 3 | *Recognition vs. Localization in the Dark: A Decomposition of Detectors and VLMs under Low-Light Enhancement* | IEEE TMM / TCSVT; NeurIPS 2027 Evaluations & Datasets (~early May) only with the artifact in §1.3 | 6/10 as a measurement paper (4/10 as the original method idea) | **Conditional** on a 1-week pilot |

This is the complete plan; splitting further would be salami-slicing. The earlier DPI-Det design and the no-regret enhancement variant are retired, and their useful parts live inside Papers 1 and 2.

---

### 1.1 Paper 1 (Q1): the study

**Safe to claim as novel** (from the full-text check, `literature/06` §3), once run with seeds and CIs:
1. The first low-light detection study to run **the same enhancers and detector families under all of P0 (zero-shot), P1 (raw→raw), P2 (raw-trained, enhanced test), P3 (enhanced→enhanced) and P4 (mixed)**, on ≥3 real datasets of different darkness, with paired-bootstrap CIs. It shows that the sign of the effect is governed by a **protocol × darkness interaction**: ExDark is not helped, DarkFace is.
2. An **operational decomposition** of the penalty into an *adaptable* part (P3 − P2) and a *residual* part (P1 − P3), controlled for augmentation order, quantization and I/O. Stratify it by object size, lighting condition and detector family. *Nobody reports P2 next to P3 for detection.*
3. **How enhancement utility scales** with training-set size (10 / 25 / 50 / 100%) and with frozen vs fine-tuned backbones. No prior work found in low light, haze, underwater or super-resolution.
4. **Controlled interventions:** rebuild each enhancer's output from luminance-matched gain/gamma only, colour only, noise residual only, and detail only. Link each to layer-wise CKA/MMD representation shift and to **per-object ΔAP**. Not done for low-light enhancement.
5. **Real-data, cross-detector** object-size effect (LR-YOLO's synthetic COCO-Dark shows the *opposite* trend, which is a selling point), and the first detection analysis over **ExDark's 10 lighting conditions**.

**Must NOT claim** (all already published):
- Enhancement hurts detection.
- Retraining on enhanced images fails to beat raw.
- Frozen-detector evaluation of enhancers.
- Protocol dependence in general.
- Quality metrics don't predict utility.
- A per-image oracle.
- A TIDE analysis.
- Small objects benefit (as a first).
- "Enhancement helps zero-shot detectors on ExDark".

**Required design**, pre-empting the criticisms of LIME-Eval (rated 3/6/8/8 at ICLR 2025, then withdrawn):
- ≥3 detector families: one-stage CNN, two-stage, and DETR.
- ≥3 real datasets spanning darkness: ExDark (MAET 80/20 split + official split), **DarkFace**, and **NOD (extreme-light subset)** or BDD100K-night.
- 3 seeds, with paired-bootstrap CIs (Holm-corrected).
- Identical tuning budget across all arms.
- **Both augmentation orders.**
- Identity-I/O, quantization and brightness-matched (gamma) controls.
- A capacity sweep (n / s / m).
- Correlation analyses over ≥50 cells.
- A "regret rate": the fraction of images or objects that end up worse than raw.

**Where the bulk of effort goes:** claims 3 and 4. They are what move this from a replication to a contribution.

---

### 1.2 Paper 2 (A\*): Distilling Computable Privilege

**One-sentence contribution:** an enhancer is a deterministic function of the raw image, so it adds *inductive bias, not information*. We estimate that bias as a **cross-fitted, seed-noise-calibrated per-object treatment effect**, distil enhanced-view teachers **only where that effect is significant**, and amortise it into a **label-free utility head** that extends the gate to unlabeled night images. The output is a plain raw-input detector with zero inference cost.

**Components:**

| Part | What | Why it is new (closest work → difference) |
|---|---|---|
| **C1, the core** | K-fold *cross-fitted* teachers (raw seeds + enhanced-domain teachers). Each object's utility z_k(o) is measured only by teachers that never saw it, and normalised by the **raw-vs-raw seed disagreement** (a null distribution). Gate = 1 only if z_k(o) > threshold; raw is the default. | M-AID and CA-MKD use **in-sample** ground-truth losses with **no noise model**, from same-domain teachers. AMIEOD picks per *image*, in-sample. TaskGuard has no low-light data and no distillation. |
| C2 | Standardised (Pearson-style) box-masked feature imitation on all pyramid levels, plus logit distillation, so the student copies structure rather than illumination statistics | A design choice, **not** a novelty claim |
| **C3** | A small head predicts z_k(o) from raw features, which allows **semi-supervised** gating on unlabeled dark images (DarkFace unlabeled set, BDD-night) | No low-light distillation paper learns a label-free utility predictor |
| Theory → testable predictions | (i) The gain shrinks as training data grows (links to Paper 1, claim 3). (ii) It concentrates on dark, small objects. (iii) Part of any distillation gain is just "ensembling", measured with raw-seed teachers through the same gate. | (iii) has not been measured in low-light distillation |

**Do NOT claim:** scale-aware gating, ground-truth-weighted teachers, or training-only enhancement. All three are taken (M-AID, CA-MKD, SGLDet).

**Recipe to maximise the chance of better results:**
1. **Base recipe = the strongest known one.** AMIEOD-style multi-view (raw + enhanced) training is the *starting point*, not just a baseline. AMIEOD's own detector gets **81.9 mAP50 on raw ExDark input** (YOLOv3, MAET split) vs 76.5 plain.
2. **Larger teachers** (YOLOv8m) distilled into a YOLOv8n/s student, with an **equally large raw teacher** as the control.
3. Box-masked feature distillation on all pyramid levels, plus logits.
4. **Datasets where enhancement utility is large:**

   | Dataset | Reported gains |
   |---|---|
   | DarkFace | SGLDet 59.7 → 76.6; YOLA 62.1 → 67.4 (TOOD) |
   | ExDark | typically +1–2 only |

   So report DarkFace and NOD-extreme as headline numbers, with ExDark as the standard benchmark.
5. Use standard splits (ExDark MAET 80/20 and the YOLA protocol), so the numbers sit next to published ones.

**Realistic targets** (from published numbers, `literature/05` §4):

| Comparison | Target |
|---|---|
| YOLOv8n over a raw model with no distillation | +2–3 mAP50, mostly from the larger teacher |
| Over the strongest zero-cost control | **+0.8–1.5 on ExDark, +1.5–3 on DarkFace** |
| The gate itself vs ungated distillation | +0.3–1.0. **This is the main risk.** |

**Mandatory controls** (same epochs and teacher size, 3 seeds, paired-bootstrap CIs on the 1,467-image MAET test set):
- no distillation; extra-epoch fine-tuning; photometric augmentation
- P4 mixed training; AMIEOD raw-input; SGLDet reimplementation
- single raw-teacher distillation; ungated multi-teacher distillation; random gate
- **in-sample M-AID-style gate**; **raw-seed-ensemble teachers through the identical gate**

**Go/no-go checkpoints** (≈550 GPU-hours on one 24 GB GPU over 10 weeks):
- **Week 2 (GNG-1):** if objects with z > 2 are no more common than under the raw-vs-raw null, utility is noise. **Stop B** and put the null result into Paper 1, where it is still publishable.
- **Week 3 (GNG-2):** the cross-fitted gate must be ≥ +0.8 over ungated distillation **and** ≥ +0.5 over the raw-seed-ensemble gate on ExDark (confirm with 2 more seeds). If not: if ungated enhanced-teacher distillation beats multi-view training by ≥ +1, reframe as "enhancement as computable privileged information" for AAAI or TCSVT. Otherwise fold into Paper 1.
- **Week 5 (GNG-3):** the label-free gate must recover ≥70% of the ground-truth gate's gain, and unlabeled data must add ≥ +1.0 on DarkFace. If not, drop C3.
- **Weeks 6–10:** third dataset, second detector (RT-DETR), AMIEOD reproduction (its code is public and Ultralytics-based), 3 seeds, data-scaling curve. Submit to ICCV 2027 only if every checkpoint passes; otherwise ACM MM 2027 / AAAI-28.

---

### 1.3 Paper 3 (conditional): Recognition vs Localization in the Dark

**Prior work it must cite and differ from:**
- Morawski et al. (TMM 2025): zero-shot YOLO-World and CLIP on ExDark/NOD/LOD × ~16 enhancers.
- VLOD-TTA (Grounding DINO / YOLO-World on ExDark, BDD).
- SIC-CADS (image-level prior rescoring, distilled into the detector).
- arXiv 2509.19875 (Qwen2-VL prior on ExDark, YOLOv12s).

**What is still open:**
- An explicit **recognition/localization decomposition**: multi-label presence mAP vs AP50/AP75/AP50-95 by object size, with CIs.
- Run across detectors (closed-set, raw-trained and COCO zero-shot, plus open-vocabulary) **and** VLMs, on ≥3 real datasets.
- A **swap test**: raw-view boxes scored with enhanced-view class evidence, and the reverse.
- A **blur/JPEG control matched on the mAP50 drop**, to show the localization fragility is specific to enhancement.

**Pilot first** (≈1 week, inference only; `literature/07` §5):

| Gate | Pass condition | If it fails |
|---|---|---|
| GO-1 | ≥2 enhancers raise presence mAP by ≥ +2 for ≥2 recognisers while AP75 does not rise | Drop Paper 3; fold the zero-shot results into Paper 1 |
| GO-2 | The matched blur/JPEG control shows a clearly smaller AP75/AP50 drop ratio | The effect is generic, not enhancement-specific |
| GO-3 | Cross-view consensus beats the best single view by ≥ +1.0 presence mAP | — |
| GO-4 | Rescoring gains ≥ +1.0 mAP50 over raw and ≥ +0.5 over a raw-only prior | — |

GO-1 + GO-2 support an analysis paper (TMM / TCSVT). All four together are needed for any method claim.

**NeurIPS Evaluations & Datasets only with an artifact:**
- exhaustive open-vocabulary presence labels for the ExDark test set plus DarkFace, BDD-night and LLVIP subsets;
- completed boxes with per-box visibility labels from two annotators;
- cached outputs of 8–10 enhancers plus controls;
- an evaluation toolkit.

---

## 2. Timeline (deadlines are estimates from 2026 cycles; confirm officially)

| When | Work | Target |
|---|---|---|
| Nov 2026 | Fix SCI (save `r`). Re-run P2/P3 on the **MAET 80/20 split** and the official split. 3 seeds. Save per-image predictions. Add DarkFace and NOD. Run the Paper 3 pilot. Run Paper 2 week-1 work: cross-fitted teachers. | — |
| Dec 2026 | Paper 2 GNG-1/GNG-2. Paper 1: protocols P0–P4, both augmentation orders, controls. | — |
| Jan–Feb 2027 | Paper 1: scaling + interventions + CKA. Paper 2 GNG-3 + full grid. | ICCV 2027 (~early Mar) only if all gates pass |
| Mar 2027 | **Submit Paper 1 to a Q1 journal.** | Pattern Recognition / TCSVT / TMM |
| by Apr 1 | **Paper 2 → ACM MM 2027.** | A\* |
| Apr–May | Paper 3 (if the pilot passed) | TMM / NeurIPS 2027 E&D (~early May) |
| Jun–Jul | Revisions; Paper 2 → AAAI-28 if rejected (~late Jul); write-up | A\* |

---

## 3. Before any experiment, in VS Code
- [ ] SCI: save `r = clamp(x/i)`, not `i`. Check `r ≥ x` holds on 20 images. Regenerate the SCI outputs.
- [ ] Zero-DCE++: the official test script **crops** to multiples of 12 from the top-left. Make sure boxes and image sizes are still consistent, or pad instead.
- [ ] Switch to the **ExDark MAET 80/20 split** (comparable to IA-YOLO, MAET, AMIEOD, SGLDet). Keep the official 3000/1800/2563 split as a second result. Drop the custom 70/20/10 split.
- [ ] 3 seeds; save per-image predictions; paired bootstrap + Holm script.
- [ ] Decide the augmentation order (enhance-first vs augment-first) and run both for P3/P4.
- [ ] Download **DarkFace** (incl. the unlabeled set) and **NOD**; add BDD100K-night later.
- [ ] Re-verify every reference by hand (remove `utm_source=chatgpt.com`; fix the LLVIP DOI).

## Files
- `literature/05_paper2_distillation_novelty_and_design.md`: Paper 2 full-text audit, redesign, numbers to beat, 10-week plan.
- `literature/06_paper1_study_novelty.md`: Paper 1 claims and must-not-claim list; verdict on the no-regret variant; LIME-Eval review record.
- `literature/07_paper3_recognition_localization_novelty.md`: Paper 3 audit and pilot design.
- `literature/01–04`: first-pass (snippet-based) audits, venues, reviewer checklist. Superseded where they conflict with 05–07.
- `analysis/`: reanalysis of the doc's numbers; `analysis/sci_bug/`: the SCI reproduction.
