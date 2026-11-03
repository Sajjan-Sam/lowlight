# Alternative directions built on our findings (as of 2026-11-03)

**How this was checked.** I verified items through web-search snippets of CVF, NeurIPS and ICLR proceedings pages and arXiv listings. Direct fetches of arxiv.org, dl.acm.org and iclr.cc were blocked by the session's network policy, and the shared web-search budget ran out near the end. Anything I could not confirm is marked **UNVERIFIED**.

---

## A. Status of the 8 directions

**1. VLM / open-vocabulary (OV) detection in low light — PARTIAL**
- VLOD-TTA (arXiv [2510.00458](https://arxiv.org/abs/2510.00458), 2025; venue UNVERIFIED). Test-time adaptation (TTA) of Grounding DINO and YOLO-World under shifts that include low light. Gains are small (about +1–2 mAP).
- "Language Prompt vs. Image Enhancement: Boosting Object Detection with CLIP in Hazy Environments" (arXiv [2604.10637](https://arxiv.org/abs/2604.10637), Apr 2026). Adds a CLIP-guided cross-entropy loss instead of an enhancer. Secondary sources report 70.42 mAP on ExDark. **This is the closest thing to "text instead of enhancement".**
- "Vision-Language Models Empowered Nighttime Object Detection with Consistency Sampler and Hallucination Feature Generator", IEEE TIP 2025. Known only from its code-repo summary, so details are UNVERIFIED. It uses CLIP to generate features for night detection.
- GEFU/SCUF ([ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_From_Enhancement_to_Understanding_Build_a_Generalized_Bridge_for_Low-light_ICCV_2025_paper.html)). A diffusion enhancer with a BLIP caption-consistency loss, evaluated on classification, detection and segmentation.
- Background: CLIP-LIT ([ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Liang_Iterative_Prompt_Learning_for_Unsupervised_Backlit_Image_Enhancement_ICCV_2023_paper.pdf), backlit enhancement), PODA ([ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Fahes_PODA_Prompt-driven_Zero-shot_Domain_Adaptation_ICCV_2023_paper.html)), CLIP the Gap (CVPR 2023, from memory) and ULDA (2024, venue UNVERIFIED). Chhipa et al. ([ECCVW 2024](https://arxiv.org/abs/2405.14874)) tested OV-detector robustness only on synthetic COCO-C/COCO-O.
- **Gap:** I found no study of zero-shot OV detectors or VLM grounders (Grounding DINO, YOLO-World, OWLv2, Florence-2, Qwen2.5-VL) on *real* low-light boxes such as ExDark × enhancers, with breakdowns by object size and lighting. I also found no work that uses VLM captions as a prior for detection in the dark.

**2. Multi-enhancer ensembles / enhancement as TTA — PARTIAL (engineering-level only)**
- Li et al. (arXiv [2405.03519](https://arxiv.org/abs/2405.03519), CVPR 2024 PBDL challenge entry). Applies several enhancements to the test images and fuses the predictions with IoU clustering (secondary sources: 0.754 vs 0.745).
- PBDL 2024 challenge report ([2406.10744](https://arxiv.org/abs/2406.10744)). A team used WBF over several detectors and found it about as good as dedicated low-light methods.
- AMIEOD ([2605.06084](https://arxiv.org/abs/2605.06084), May 2026). Multiple enhancement *experts* (a frozen SCI model plus learned modules). A selector picks **one expert per image** at inference, trained with a detection-guided CE loss. Reports ExDark 82.1 vs 76.5 (YOLOv3). It is UNVERIFIED whether "no enhancement" is one of the options.
- **Caption consensus across enhanced views: OPEN.** No hits. The closest is Test-Time Consistency in VLMs ([WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Chou_Test-Time_Consistency_in_Vision_Language_Models_WACV_2026_paper.html)), which enforces agreement across semantically equivalent inputs for VQA, not enhancement or detection.
- **Gap:** No one has learned *per-box* fusion of raw + K enhanced views, and no one has explained why ensembling enhancers helps when each one alone hurts.

**3. Scale-aware enhancement / raw + enhanced dual-branch fusion — fusion DONE; scale-aware OPEN**
- Fusion of raw and enhanced features already exists: DBLDNet (Multimedia Systems 2025), NLE-YOLO ([Sci. Rep. 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10891144/)), WTEFNet ([2505.23201](https://arxiv.org/abs/2505.23201)) and DBM-YOLO (Drones 2025). All fuse the same way at every scale.
- I found no paper that reports or exploits the *size-dependent sign* of the enhancement effect, i.e. small objects gaining while medium and large objects lose.

**4. Distillation: teacher on enhanced/bright images, student on dark — PARTIAL**
- Bright-to-dark feature distillation with no test-time cost: Jaw et al., RMD-Net ([IEEE TCyb](https://dmas.lab.mcgill.ca/fung/pub/JHLFK23cyb_preprint.pdf)).
- Clean Feature Distillation for Low-Light Detection ([Springer LNCS chapter](https://link.springer.com/chapter/10.1007/978-981-95-5755-4_9); venue UNVERIFIED).
- KDFA ([Sensors 2025](https://www.mdpi.com/1424-8220/25/15/4871)), daylight teacher for apple detection.
- **Strongest competitor: SGLDet ([ICLR 2026](https://mlanthology.org/iclr/2026/shin2026iclr-selfguided/)).** A train-only auxiliary enhancement + denoising + Fourier-fusion pipeline with zero inference cost. Reported gains on DARK FACE (76.6 mAP), ExDark and nuImages.
- TorchAdapt ([ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Hashmi_TorchAdapt_Towards_Light-Agnostic_Real-Time_Visual_Perception_ICCV_2025_paper.html)) aligns features of enhanced and well-lit images.
- **Per-object gated / utility-weighted distillation: OPEN.** The nearest is "Adaptive Instance Distillation", which weights instances by teacher confidence, uses a single teacher and is not about low light.

**5. TTA for night / low-light detection — PARTIAL**
- [NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/88c3c482430a62d35e03926a22e4b67e-Abstract-Conference.html): TTA for Grounding DINO with multimodal prompts and a mean teacher. Benchmarks are cross-corruption and cross-dataset.
- Yoo et al. ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yoo_What_How_and_When_Should_Object_Detectors_Update_in_Continually_CVPR_2024_paper.html)): continual TTA on SHIFT, whose domain sequence includes dawn and night.
- BN-statistics TTA for driving ([2411.18860](https://arxiv.org/abs/2411.18860); AAAI version, year UNVERIFIED): synthetic low light only. Re-estimating BN statistics beats TENT.
- Enhancer + TTA: an ICCV 2023 VCL challenge report ([2310.08986](https://arxiv.org/abs/2310.08986)), and TECA ([2403.17423](https://arxiv.org/abs/2403.17423)), which switches between raw and enhanced logits by uncertainty. TECA is **classification only**; venue UNVERIFIED.
- **Gap:** No TTA work on *real* low-light detection benchmarks (ExDark, BDD-night), and no detection version of TECA's raw/enhanced switching.

**6. Learning when to enhance / predicting utility — PARTIAL and filling quickly (most urgent)**
- **TaskGuard ([2609.08011](https://arxiv.org/abs/2609.08011), Sep 2026).** Predicts whether a restored image is useful for detection and accepts or rejects it. On DAWN rain it restored only 8 of 198 images yet kept about 78% of the AP gain from always restoring. Its low-light results are UNVERIFIED (I did not see any).
- PreActIR ([2610.20384](https://arxiv.org/abs/2610.20384), Oct 2026). Predicts a restoration's effect before applying it, but on image-quality targets, not detection.
- LIME-Eval ([2410.08810](https://arxiv.org/abs/2410.08810)). A label-free detector-energy score with Spearman 0.881 against mAP.
- AMIEOD (per-image expert choice) and TorchAdapt (content-adaptive modulation at feature level).
- **Gap:** Nobody does per-object or per-region, size-aware utility prediction over a pool of *off-the-shelf* enhancers on real low-light data, with "keep raw" as the default.

**7. Diffusion priors / generative enhancement for detection — PARTIAL and crowded**
- LightDiff ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Light_the_Night_A_Multi-Condition_Diffusion_Framework_for_Unpaired_Low-Light_CVPR_2024_paper.html)): conditions on depth, RGB and text captions, with a reinforcement-learning reward from 3D detectors on nuScenes night.
- GEFU (ICCV 2025), EDTR ([ICCV 2025](https://arxiv.org/abs/2507.22459)) for task-driven restoration, and Contrastive-SDXL ([2605.16406](https://arxiv.org/abs/2605.16406)) for day-to-night augmentation.
- Not advised: this area is crowded and heavy on compute for a single GPU.

**8. Low-light benchmarks for VLMs / MLLMs — PARTIAL (VQA covered; captioning and grounding with boxes OPEN)**
- DarkQA/DarkEQA ([2512.24985](https://arxiv.org/abs/2512.24985); RA-L 2026 per its project page). Synthetic indoor low light from HM3D. LLIE gives *"severity-dependent but unstable recovery"*, and at the worst level some VLMs score below a blind LLM. **This independently mirrors our finding, but for VQA.**
- EgoNight ([ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/022ca1bed6b574b962c48a2856eb207b-Abstract-Conference.html)): night egocentric video QA. Best score is GPT-4.1 at 30.9%.
- Also: R-Bench ([2410.05474](https://arxiv.org/abs/2410.05474), luminance among 33 corruptions) and ObsDriveBench ([2607.23537](https://arxiv.org/abs/2607.23537), adverse weather including low light).
- **No real low-light captioning or grounding benchmark with boxes was found.**

---

## B. Proposed method ideas

### Idea 1 — Per-box, size-aware utility fusion of raw and enhanced views ("Enhance-or-Not")
- **Insight:** Whether enhancement helps is *predictable per object*, from box size, local brightness and noise, the enhancer used, and agreement across views. So fusion should decide per box, not per image.
- **Method:**
  1. Run the frozen raw-trained detector on the raw image and on the 7 enhanced views. The P2 outputs already exist.
  2. On the validation set, label each predicted box as a true or false positive.
  3. Train a small MLP or GBDT on: log area, local mean intensity, an SNR estimate, the enhancer's ID, the box's confidence in each view, and the IoU agreement between views.
  4. At test time, use the predicted utility as the weight in a weighted box fusion (WBF) step, with the raw view as a prior.
  5. Optional: a cheap image-level "skip" gate that predicts per-image ΔAP before running any enhancer.
- **Closest work and difference:**
  - AMIEOD picks one jointly trained expert per image.
  - TaskGuard accepts or rejects a whole image for a single restorer, and its low-light experiments are UNVERIFIED.
  - 2405.03519 uses heuristic clustering in a challenge entry.
  - None of them works per box, conditions on object size, or pools off-the-shelf enhancers.
- **Feasibility:** Very high. No detector retraining is needed, and it runs on CPU or one GPU. About 2–3 months, including DarkFace and BDD-night.
- **Venue:** A Q1 journal (Pattern Recognition, IEEE TCSVT). A* is possible if combined with Idea 3.
- **Risk:** Reviewers may call it "just ensembling", and gains may be small. Compare at equal compute against flip and multi-scale TTA and against an ensemble of raw models. TaskGuard may add low-light experiments, so move quickly.

### Idea 2 — Size-routed raw/enhanced feature fusion
- **Insight:** Small objects gain from enhancement and large ones lose, so inject enhanced features **only at the high-resolution FPN level** (P3), or use a learned gate per FPN level conditioned on an illumination/SNR map.
- **Method:** A shared-weight (Siamese) backbone on raw and LIME or Zero-DCE++ inputs (both cheap). Gates per level, α_P3 > α_P4 > α_P5, initialised from the measured per-size ΔAP. Optionally a box-size-conditioned gate in the head.
- **Closest work and difference:** DBLDNet, NLE-YOLO and WTEFNet fuse the same way at every scale and have no size hypothesis.
- **Feasibility:** High with YOLOv8n in Ultralytics (about 1 month).
- **Venue:** Q1 journal.
- **Risk:** Doubles backbone compute, and the gain may simply come from adding more parameters. A control with the same parameter count is needed.

### Idea 3 — Utility-gated distillation from enhanced views, with a raw-only detector at inference (strongest A* candidate)
- **Insight:** Enhancement hurts on average but helps a predictable subset of objects (small or very dark ones). So distill **only on the objects where an enhanced view actually beats the raw view**, decided with ground truth at training time.
- **Method:**
  1. Teachers are the raw-trained detector on each enhanced view, plus the already-trained P3 YOLOv8n models.
  2. For each ground-truth object, compute the utility of each view (IoU × confidence), and set the gate g_k = softmax over views of (utility_k − utility_raw), clipped at zero.
  3. Apply feature distillation (masked to the object's box) and logit distillation from the gated teacher. Where no view beats raw, use self-distillation.
  4. Inference uses raw input only, with no enhancer.
  5. Optional: learn the gate from Idea 1's features so it also works without ground truth, e.g. on unlabeled night data.
- **Closest work and difference:**
  - RMD-Net, CFD and KDFA distill from a bright teacher everywhere, with no gate.
  - SGLDet (ICLR 2026) uses an auxiliary enhancement branch only at training time, also ungated.
  - Adaptive Instance Distillation weights by confidence within a single domain.
  - TorchAdapt aligns all features.
- **Feasibility:** Moderate. It needs distillation hooks in Ultralytics or MMDetection; about 3–4 months. Baselines: YOLA, FeatEnHancer, SGLDet, FRBNet ([NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/8ea50bf458f6070548b11babbe0bf89b-Abstract-Conference.html)).
- **Venue:** A* (ACM MM, AAAI, ICCV 2027) if it holds on 3 or more datasets.
- **Risk:** The raw ExDark baseline is strong, so gains may be under 1 mAP. Mitigate by testing cross-dataset (ExDark to DarkFace or BDD-night), where gating should matter more.

### Idea 4 — "Dark-Ground": a benchmark plus analysis of foundation detectors and VLMs × enhancers
- **Insight:** Test whether "enhancement hurts, and the effect depends on object size" also holds for open-vocabulary detectors and VLM grounders. DarkQA hints that it does for VQA.
- **Method:**
  1. Evaluate Grounding DINO, YOLO-World, OWLv2, Florence-2 and Qwen2.5-VL-3B/7B zero-shot.
  2. Datasets: ExDark, DarkFace, BDD-night and LLVIP-visible, each × 7 enhancers.
  3. Break results down by size, by ExDark's 10 lighting conditions and by image-quality metrics, and correlate with ΔAP.
  4. Add a simple fix: Idea C below, and/or VLOD-TTA.
- **Closest work and difference:**
  - DarkQA: synthetic, indoor, VQA, no boxes.
  - EgoNight: video QA.
  - Chhipa et al.: synthetic corruptions only.
- **Feasibility:** High (inference only), about 2 months.
- **Venue:** NeurIPS Datasets & Benchmarks track or ACM MM (A*), or a Q1 journal.
- **Risk:** Benchmark-only papers need breadth and new annotations. Mitigate by adding a small manually captioned subset of 500 ExDark images.

---

## C. Assessment of the whiteboard idea (enhancers → captions → text-guided detection)

- **Novelty:** I found no paper with this exact pipeline. Its parts all exist:
  - VLM captions used to *train* detectors: LLMDet (CVPR 2025), Pseudo Caption Labeling, and synthetic-caption hyperbolic learning ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Kong_Hyperbolic_Learning_with_Synthetic_Captions_for_Open-World_Detection_CVPR_2024_paper.pdf)).
  - Captions or text guiding *enhancement*: GEFU (ICCV 2025), LightDiff (CVPR 2024), NaLSuper ([2501.06546](https://arxiv.org/abs/2501.06546)).
  - Language prompts in place of enhancement for degraded detection: 2604.10637, and the TIP 2025 CLIP night detector.
  - Consensus across views at test time: WACV 2026.
- **Problems with the naive version:**
  - Captions are image-level and give no localization.
  - Captioners hallucinate in the dark. DarkQA shows VLMs falling below a blind LLM at the worst levels.
  - ExDark has only 12 closed-set classes, so text adds little to a YOLO model beyond class-presence information.
  - Running K enhancers plus K VLM calls per image is expensive.
- **Strongest version ("enhance for *what*, not *where*"):**
  - Hypothesis: enhanced views improve *recognising which objects are present* even though they hurt *localization*. This fits our finding that localization drops while some recognition survives.
  - Step 1: Run a VLM on the raw view and K enhanced views to get object lists.
  - Step 2: Take a **reliability-weighted consensus vote**, with weights learned on the validation set. This turns many noisy captions into a class-presence probability with fewer hallucinations.
  - Step 3: Use that probability to (a) build the prompt for Grounding DINO or YOLO-World, querying only classes that are present, which cuts prompt-induced false positives, and (b) rescale the scores of the raw closed-set detector's boxes.
  - Step 4 (cheap deployment): distill the consensus into an image-level multi-label head on the raw detector, so inference needs no VLM. This links to Idea 3.
- **Worth pursuing?** Yes, but as a **secondary contribution** and only after a 1–2 week pilot. On the 737 ExDark test images × 8 views, measure two things: (i) class-presence accuracy (multi-label mAP) from the VLM and from the detector, and (ii) the hallucination rate of each view against the consensus. If enhanced views improve presence recognition while box AP falls, the "what vs. where" story is new and publishable (pairs with Idea 4 for ACM MM or NeurIPS D&B). If not, drop it, because the naive pipeline will be slow and weak. Main competitors to cite and beat: 2604.10637 (CLIP-CE on ExDark) and the TIP 2025 CLIP night detector.

**Overall recommendation:** Ideas 1 and 3 share the same per-object utility signal and form one coherent story: "enhancement utility is predictable per object: fuse it at test time, or distill it with no enhancer at inference." Idea 4 plus Idea C is a parallel benchmark paper. Before submitting any of them, the student needs at least 3 datasets, 3 seeds, and must fix the possible SCI pipeline bug. Approximate deadlines (UNVERIFIED): ICCV 2027 around March 2027, ACM MM 2027 around April 2027, NeurIPS 2027 around May 2027.
