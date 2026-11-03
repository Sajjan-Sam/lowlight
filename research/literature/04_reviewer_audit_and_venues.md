# Reviewer audit and venue strategy: "Does LLIE help detection?" (2026-11-03)

**Verdict (area chair):** in its current state this is a reject at A* and at best a major revision at a Q1 journal.
- **Two potentially fatal problems, cheap to fix in 2–3 weeks:** the enhancer pipeline is unverified (SCI looks broken, §1.2), and there are no variance estimates.
- **The scientific weakness:** "enhancement hurts" under P2 is expected by construction. It becomes a finding only with a normal-light-detector protocol (P0) and causal controls.

## 1. Methodology audit

### 1.1 Issues (F = fatal, M = major, m = minor)
| # | Issue | Sev | Fix |
|---|---|---|---|
| 1 | **One seed, no CIs, no tests.** The noise is visible in the doc: YOLOv8n P1 is reported as both .7318 and .7332. LIME is +.031 (RT-DETR) on the 74-image subset but −.003 on the full test set. Estimate: seed SD ≈ .003–.01 and paired-bootstrap CI ≈ ±.005–.015. So LIME's Δ (−.005…−.012), LIME+FRCNN (+.004) and the P3 gaps for LIME/Zero-DCE++ (−.002/−.004) are almost certainly *n.s.* The −.09…−.15 drops probably are significant, if the pipeline is correct. | F | 3 seeds (5 for headline numbers). Paired image-level bootstrap CI on every Δ. Holm correction over the 35 cells. |
| 2 | **The pipeline is unverified**: SCI, resizing, EXIF orientation, JPEG re-encoding. The doc's "Next step: Image Registration" hints that misalignment was already seen. | F | Run the §1.2 checks. Add an **identity-enhancer control**: raw images through the same read/save path must reproduce P1. |
| 3 | **P2 is domain shift by construction**, because the detectors were fine-tuned on dark ExDark. Reviewers will call it a strawman, and 2409.14461 and LIME-Eval already report harm, so novelty is thin. | F | Add **P0**: COCO-pretrained detectors, zero-shot, on raw vs. enhanced images. All 12 ExDark classes exist in COCO, so this is where LLIE *should* help. Frame P2 as a test of the "plug-and-play" claim. Add P4 (mixed training) and an oracle per-image best-of{raw, enh}. Rebut LIME-Eval's critique of retrained detectors explicitly. |
| 4 | Only one dataset, yet Objective 1 claims "generalization". | M | Add DarkFace and LLVIP-visible (or BDD100K-night). |
| 5 | **Custom 70/20/10 split**, so no comparability. ExDark's `imageclasslist.txt` has its own split (3,000/1,800/2,563, UNVERIFIED). MAET, IAT and YOLA use an 80/20 per-class split (≈5.9k/1.47k). | M | Re-run P1/P2 on the MAET split. Reproduce one published number. Release the split files. |
| 6 | **P3 covers only YOLOv8n and 5 of 7 enhancers.** PairLIE and KinD++, the two worst in P2, are missing, which looks selective. | M | All enhancers × ≥3 detectors × 3 seeds. |
| 7 | **Enhancer checkpoints are unreported.** SCI has easy/medium/difficult weights; Retinexformer and SNR-Aware ship LOL-v1/v2/SID/SMID weights. | M | State the selection rule. Sensitivity over ≥2 checkpoints. |
| 8 | **Mixed evaluators**: Ultralytics vs. pycocotools, with P/R at different confidence thresholds. | M | Score all detectors with pycocotools from COCO JSON. Add a TIDE error breakdown. |
| 9 | **Size analysis**: COCO's absolute 32²/96² thresholds on variable-resolution images ("640×360" is one image). AP_S = .126 rests on few instances, so "small objects helped" is likely noise. | M | Relative-area bins, instance counts per bin, CIs. |
| 10 | Per-class ΔAP has no instance counts. Classes are imbalanced, and rare classes dominate the variance of a class-averaged mAP. | M | Report counts and CIs. Stratified split. |
| 11 | **The "why" is untested**: no IQA, no breakdown by ExDark's 10 lighting conditions, no mechanism tests. A correlation over n = 7 enhancers is meaningless. | M | Brightness-matched control (global gain to each enhancer's mean), noise injection, edge/frequency energy, TIDE. Mixed-effects regression of image-level ΔTP/ΔFP on IQA. |
| 12 | **Detector set**: v8n ≈ v10n is redundant; nano vs. large confounds capacity; Deformable DETR (.67) looks under-trained; no schedules reported; no open-vocabulary detector. | M | Report schedules. Use non-nano models. Add Grounding-DINO or YOLO-World zero-shot. |
| 13 | **Annotation noise**: loose or missing ExDark boxes. Enhancement reveals unlabeled objects, which count as false positives (FPs); precision falls .81→.70 under SCI. | M | Audit 200 top FPs per enhancer. Optionally re-check the 737 test labels (~2 days), which would itself be a contribution. |
| 14 | **Leakage**: were enhancer or weight choices made on the test set? No duplicate check across splits or against COCO train, which matters because the detectors are COCO-pretrained. | m→M | Confirm `val≠test` in data.yaml. Run pHash/CLIP dedup. |
| 15 | The 10%-subsample table contradicts the full-test results. No pipeline latency or FLOPs reported. | m | Drop the table. Report FPS/FLOPs. |

### 1.2 SCI: strong evidence of a bug
- **The official code guarantees brightening.** `vis-opt-group/SCI/CVPR/model.py` (read today) computes `i = clamp(x+σ(f(x)),1e-4,1)` and `r = clamp(x/i,0,1)`, and `test.py` saves `r`. So **r ≥ x at every pixel, with local gain ≥ 1.** A correct SCI output can never be darker or lower-contrast than its input.
- **The doc's figures violate this.** I measured them (500×333): fitting SCI vs. raw gives **slope 0.24–0.80 and intercept ≈ 35–44** (a grey veil), and **mean |gradient| drops by about 40%** (2.78→1.67 and 3.22→2.06). In one scene, bright regions darken (186→160). Every other enhancer has slope ≥ 1 and stronger gradients than raw.
- **Likely cause:** the smooth illumination map `i` (or a wrong tensor or normalization) was saved instead of `r`.
- **Other notes from the repo:**
  - The loader splits paths on `'\\'` (Windows-only), so the student must have edited the I/O code; audit those edits.
  - The default weights are `medium.pt` (LOL+LSRW).
  - The README says the released weights did not produce the paper's results.
- **Checks:**
  - (a) Run the unmodified repo on its sample data and diff against the student's outputs.
  - (b) Assert `enh ≥ raw−2/255` on >99% of pixels.
  - (c) Assert enhanced and raw shapes match, for all enhancers.
  - (d) Check RGB vs. BGR and the [0,1] input range.
  - (e) Check EXIF orientation: cv2 applies it, PIL does not.
  - (f) Overlay GT boxes on 20 outputs per enhancer.
  - (g) Run the identity control.
- **Consequence:** if SCI was buggy, its headline −.149 is invalid.

### 1.3 Factual errors in thesis.md
1. The LLVIP citation links to LIME's DOI 10.1109/TIP.2016.2639450. The correct source is arXiv 2108.10831 (ICCVW 2021); its 15,488-pair count is verified.
2. ExDark "640×360, mean 9.84" are stats from one image; ExDark resolutions vary.
3. The LoL-v2 row claims "real + synthetic", but its counts and link cover only the Real subset.
4. The "KAIST / FLLIR" row describes only FLIR (via a Kaggle mirror); KAIST is a different dataset.
5. Dark Zurich, merged with NightOwls, is a segmentation dataset with no boxes.
6. BDD100K is 1280×720, not "high-resolution"; instance masks exist only for a 10K subset. Use the official release.
7. ACDC is said to have detection labels in one table and only panoptic labels in another (UNVERIFIED).
8. YOLOv8n P1 appears with two different values.
9. Faster R-CNN + LIME has mAP50-95 = .5150, identical to raw; check for a copy-paste error.
10. "Boxes stay valid because the transform is pixel-wise" holds only if size and orientation are preserved.
11. Every citation carries `?utm_source=chatgpt.com`, i.e., LLM-sourced references. Verify each and cite the actual venues.

AMIEOD (arXiv 2605.06084) exists; verified.

### 1.4 Problems with the proposed DPI-Det method
| Issue | Sev | Fix |
|---|---|---|
| **Object masks** come from GT boxes in training but from proposals at test. This is a train–test mismatch, and box-shaped brightening can leak labels. | F | Train on predicted masks plus mask dropout. Report the oracle-mask result only as an upper bound. |
| **Kitchen sink**: 4 modules, 8 losses, 7 λ weights. No single insight, and the ablations explode. | M | Keep ≤2 modules and ≤3 losses, each tied to a benchmark finding. |
| **The losses favour doing nothing.** L_id, L_feat and L_freq all anchor I_dpi to I_raw, and raw is already best, so W→1 is near-optimal and gains may come only from the FiLM adapter or extra fine-tuning. | M | Parameter- and schedule-matched control without the illumination branch. |
| **The Poisson-Gaussian SNR is applied to JPEG sRGB** (gamma, denoising, correlated noise, 8-bit quantization), with α and β learnable, so no physics remains. | M | Call it a "learned noise-confidence map". Alternatively, unprocess to linear RGB (Brooks, CVPR'19) and validate on RAW data (e.g., LIS, IJCV'23, UNVERIFIED). |
| **The Retinex split is unidentifiable without pairs** (trivial solution R=I, L=1). | M | Add priors, or drop the claim. |
| **Novelty** must be shown against YOLA (NeurIPS'24), FeatEnHancer (ICCV'23), MAET, IAT, DAI-Net (CVPR'24) and AMIEOD. | M | Compare head-to-head on their protocols. |
| The captioning idea is unrelated to the evidence so far. | m | Treat it as future work. |

## 2. Venues

### 2.1 Conferences
Ranks are CORE2023, the latest round I could verify. A 2026 ICORE round may exist (UNVERIFIED).

| Venue | CORE | Next deadline | Fit |
|---|---|---|---|
| CVPR 2027 | A* | registration Nov 10, paper **Nov 16, 2026** (official CFP, via search) | Not feasible |
| ICCV 2027 | A* | ≈ Mar 1–8, 2027 (ESTIMATE) | Only if the method is ready by Feb |
| ECCV | A* | ≈ Mar 2028 (ESTIMATE) | — |
| NeurIPS 2027 | A* | ≈ early May 2027 (ESTIMATE; 2026 deadlines were May 4/6). The D&B track is now **"Evaluations & Datasets"** | A* home for the benchmark |
| ICLR | A* | 2027 deadline passed (Sep 25, 2026); 2028 ≈ late Sep 2027 (ESTIMATE) | Weak fit |
| AAAI-28 | A* | ≈ late Jul 2027 (ESTIMATE; AAAI-27 deadlines were Jul 21/28, 2026) | **Primary target for the method** |
| IJCAI 2027 | A* | ≈ mid-Jan 2027 (ESTIMATE) | Too early |
| ACM MM 2027 | A* | ≈ Apr 1, 2027 (ESTIMATE; 2026 deadlines were Mar 25/Apr 1; has a Dataset track) | Method or benchmark |
| WACV 2028 / BMVC 2027 | A / A | ≈ Jul 2027 / ≈ May 2027 (ESTIMATE) | Fallbacks |
| ICIP 2027 / ICASSP 2028 | B / B | ≈ Jan–Feb 2027 / ≈ Sep 2027 (ESTIMATE) | Not A* |

**Plan:**
- **Benchmark:** fix the issues above, then submit to a Q1 journal by about Mar 2027. Try NeurIPS'27 Evaluations & Datasets first only if a released toolkit and a cleaned ExDark test set make it a real benchmark.
- **Method:** ACM MM'27 if ready by March, otherwise **AAAI-28**.
- **Skip CVPR'27.**

### 2.2 Q1 journals
| Journal | Q1 | Takes benchmark-only papers? | 1st decision (ESTIMATE) |
|---|---|---|---|
| TPAMI / IJCV | Q1 | TPAMI rarely; **IJCV yes** (e.g., VE-LOL benchmark, 2021) | 4–8 mo |
| TIP / TMM | Q1 | Rarely | 3–6 mo |
| TCSVT | Q1 | Sometimes | 2–4 mo |
| TITS | Q1 | Only with a driving framing | 3–6 mo |
| Pattern Recognition | Q1 | Yes, if insight-driven | 2–4 mo |
| EAAI | Q1 (IF 8.0)* | **Yes** | 1.5–3 mo |
| Neurocomputing | Q1 (IF 6.5)* | Yes | 2–4 mo |
| Image & Vision Computing | Q1 (IF 4.2)* | **Yes** | 2–3 mo |
| ESWA / KBS | Q1 | ESWA yes; KBS leans to methods | 1.5–3 mo, high desk-reject rate |
| Information Fusion | Q1 | No (out of scope) | — |
| JVCIR | **Not Q1** (Q2/Q3, IF 2.8)* | — | — |

\*2025 JCR figures from the journalmetrics.org aggregator. The others are long-standing Q1 but were not re-checked today. Confirm whether IISER uses JCR or SJR.

**Best fits:**
- **Benchmark:** IJCV (reach), PR, IVC, EAAI, Neurocomputing.
- **Method:** TCSVT, TIP (reach), PR, TMM.

## 3. Reviewer-proof checklist
All compute figures are ESTIMATES for 1× RTX 3090/4090. Per run on ~5k training images: YOLOv8s ≈ 1.5 h, Faster R-CNN (24 epochs) ≈ 2.5 h, RT-DETR-R50 ≈ 5 h, DINO / Deformable DETR ≈ 8 h. Enhancement takes 0.05–0.5 s per image (diffusion 5–20 s).

### (a) Q1 benchmark paper
- **Data:** ExDark (MAET split plus the custom split), DarkFace, LLVIP-visible (BDD100K-night optional).
- **Detectors:** three families (YOLOv8s/11s, Faster R-CNN, RT-DETR or DINO) plus one zero-shot open-vocabulary detector.
- **Enhancers:** 9–10:
  - trivial baselines: gamma, CLAHE
  - LIME, Zero-DCE++, SCI (fixed)
  - RUAS or EnlightenGAN, PairLIE
  - KinD++, Retinexformer, SNR-Aware
  - one diffusion model and one 2025 method
- **Protocols and controls:** P0–P4, oracle best-of, identity control, brightness-matched control.
- **Analyses:**
  - per-class, relative-size and per-lighting-condition breakdowns
  - TIDE error breakdown
  - NR-IQA (NIQE, BRISQUE, MUSIQ/CLIP-IQA) and LOL fidelity (PSNR, SSIM, LPIPS)
  - mixed-effects regression of image-level ΔTP/ΔFP
  - FP audit
- **Statistics:**
  - 3 seeds, paired-bootstrap 95% CIs, Holm correction
  - Friedman test with a Nemenyi critical-difference diagram over detector×dataset blocks
  - Kendall's W for agreement between enhancer rankings
- **Compute:**
  - P1 ≈ 140 h
  - P3+P4 on ExDark (11 settings × 3 detectors × 3 seeds) ≈ 300 h
  - P3 on a second dataset ≈ 100 h
  - inference and IQA ≈ 70 h
  - **Total ≈ 600 GPU-h** (lean version ≈ 300).

### (b) A* method paper
- **Protocols, for comparability:**
  - ExDark-MAET split with YOLOv3 (the IAT/MAET/YOLA setting) and TOOD, plus one modern detector
  - DarkFace (UG2+)
  - one driving or pedestrian set
  - zero-shot COCO→ExDark/DarkFace (DAI-Net style)
- **Baselines:** re-run at least 4 under identical settings. Candidates:
  - raw, best LLIE with P3, and P4
  - MAET, IAT, PE-YOLO or DENet
  - FeatEnHancer, YOLA
  - DAI-Net, AMIEOD
  - the newest 2025–26 A* work (needs a fresh literature search)
- **Required experiments:**
  - component ablations (3 seeds)
  - the parameter-matched control
  - oracle vs. predicted masks
  - λ-sensitivity
  - Params, FLOPs and FPS
  - failure cases
  - significance tests on the main table
- **Expected bar:** reviewers will likely expect ≥ +1.5–2 mAP50 over the strongest baseline on ≥2 datasets (ESTIMATE).
- **Compute:**
  - development ≈ 250 h
  - final runs ≈ 110 h
  - ablations ≈ 180 h
  - baselines ≈ 150 h
  - **Total ≈ 700 GPU-h**: about a month running 24/7, so plan for two.

**Sources:**
- cvpr.thecvf.com 2027 CFP (via search; the host is blocked here)
- neurips.cc/Conferences/2026/Dates
- aaai.org (AAAI-27 call)
- iclr.cc (2027 CFP)
- 2026.acmmm.org (CFP)
- portal.core.edu.au entries (via search)
- journalmetrics.org
- github.com/vis-opt-group/SCI (cloned)
- arXiv 2205.03346 (MAET), 2410.18398 (YOLA), 2108.10831 (LLVIP), 2605.06084 (AMIEOD)
