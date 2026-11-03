# Paper 1 Execution Plan: When Does Low-Light Enhancement Help Detection? (Protocol × Darkness Study with Controlled Interventions)

Research date: 2026-11-03. Scope: a concrete "run model M on dataset D" plan that one student can execute in VS Code on one 24 GB GPU, targeting Pattern Recognition, IEEE TCSVT or IEEE TMM.

**How these notes were verified.**
- I cloned or read raw files from these GitHub repos: ExDark, MAET, IAT, YOLA, NOD, SCI, Zero-DCE++, Retinexformer, PairLIE, KinD++, LIME-python, LLVIP, TIDE, autorank, Ultralytics and MMDetection.
- I computed all counts marked "computed" myself from the downloaded annotation files (ExDark `imageclasslist.txt`, NOD COCO JSONs).
- Several hosts could not be reached from this session: arxiv.org, flyywh.github.io (DarkFace), doc.bdd100k.com and ieee-cas.org. Facts that depend on them come from search snippets or from the project's own `research/literature/05_*.md`, and are marked as such.
- Anything not checked is marked **UNVERIFIED**.
- **GPU-hour figures are my estimates, not measurements.** Calibrate them by timing one epoch of each detector first (step 0.6 below).

Project shorthand used throughout:
- **P0**: normal-light-pretrained detector, zero-shot on raw vs enhanced images.
- **P1**: train on raw, test on raw.
- **P2**: train on raw, test on enhanced.
- **P3**: train and test on enhanced.
- **P4**: train on a raw+enhanced mix, test on raw and on enhanced.
- **Adaptable penalty** = P3 − P2. **Residual penalty** = P1 − P3.

---

## Q1. Datasets: sources, licences, sizes, formats, standard splits, COCO mappings

### Takeaway
Use these three datasets as the must-have:
- **ExDark with the MAET 80/20 split** (5,896/1,467). Its split files ship inside the MAET/IAT Google Drive archive, and the official 3000/1800/2563 split comes from `imageclasslist.txt`.
- **DarkFace with the YOLA 5,400/600 split.** YOLA's split files are on Google Drive. There is no single community standard: FRBNet uses 3:1:1 and SGLDet uses 5,500/500.
- **NOD.** The official COCO JSONs use an 80/10/10 split, and instance-level `isextreme` flags exist only for the 321-image Sony test set.

BDD100K-night (`timeofday == "night"`) and LLVIP-visible are nice-to-have.

The ExDark filenames are ordered by class. Any "first-k" subsetting, including Ultralytics `fraction`, therefore produces a class-biased subset.

### Cited Findings

**ExDark (Exclusively Dark)**
- 7,363 low-light images, 12 classes, 10 lighting conditions "from very low-light to twilight". Repository code is under BSD-3-Clause. — [ExDark repo](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)
- Images: a 1.5 GB Google Drive archive, one folder per class.
  - Per-class counts: Bicycle 652, Boat 679, Bottle 547, Bus 527, Car 638, Cat 735, Chair 648, Cup 519, Dog 801, Motorbike 503, People 609, Table 505.
  - The paper's training split is 3,000 images (250 per class).
  - [Dataset/README.md](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Dataset/README.md); image download: https://drive.google.com/file/d/1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx/view
- Ground truth: a 4.2 MB Google Drive archive, one `.txt` per image in per-class folders.
  - Format: Piotr's Toolbox bbGt. The first 16 characters are unused; then each object is `class l t w h` followed by 7 unused occlusion/orientation columns.
  - [Groundtruth/README.md](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Groundtruth/README.md); GT download: https://drive.google.com/file/d/1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i/view
- `imageclasslist.txt` columns: name | class (1–12) | lighting type | indoor(1)/outdoor(2) | split (Training 1, Validation 2, Testing 3).
  - Lighting codes: Low 1, Ambient 2, Object 3, Single 4, Weak 5, Strong 6, Screen 7, Window 8, Shadow 9, Twilight 10.
  - [Groundtruth/README.md](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Groundtruth/README.md)
- Computed from [imageclasslist.txt](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Groundtruth/imageclasslist.txt):
  - The official split is exactly 3,000 / 1,800 / 2,563.
  - Lighting counts over all 7,363 images: Low 241, Ambient 2,098, Object 498, Single 762, Weak 705, Strong 2,059, Screen 165, Window 363, Shadow 56, Twilight 416.
  - Lighting counts in the official test set: 134 / 925 / 144 / 255 / 232 / 538 / 28 / 119 / 14 / 174.
  - Indoor 2,822, outdoor 4,541.
  - File extensions are mixed: 6,195 .jpg, 496 .png, 424 .JPEG, 243 .JPG, 5 .jpeg.
  - Files are numbered in class order: the first 736 sorted filenames are 652 Bicycle + 84 Boat.
- Licence (new "Research Use Terms v1.0", effective 2026-10-01): annotations, metadata and splits are licensed for **non-commercial academic research, education, benchmarking and evaluation**. Underlying third-party photographs are excluded from the grant. Aggregate results may be published. — [Dataset/DATASET_TERMS.md](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/Dataset/DATASET_TERMS.md)
- **MAET split.**
  - MAET distributes ExDark "in VOC format" from Google Drive (`1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC`) or Baidu (password 1234). The layout is `JPEGImages/IMGS` (plus pre-enhanced `IMGS_Kind`, `IMGS_ZeroDCE`, `IMGS_MEBBLN`), `Annotations/`, `main/`, `label/`.
  - The MAET YOLOv3 config trains on `main/train.txt` and uses `main/val.txt` for **both val and test**.
  - Recipe: 608×608 test scale; multi-scale 320–608 training; SGD lr 0.001; 24 epochs (steps 18/23); 8 images/GPU.
  - Classes: `('Bicycle','Boat','Bottle','Bus','Car','Cat','Chair','Cup','Dog','Motorbike','People','Table')`.
  - Sources: [ICCV_MAET repo](https://github.com/cuiziteng/ICCV_MAET); [maet_yolo_exdark.py](https://github.com/cuiziteng/ICCV_MAET/blob/main/configs/MAET_yolo/maet_yolo_exdark.py); [Exdark.py](https://github.com/cuiziteng/ICCV_MAET/blob/main/mmdet/datasets/Exdark.py)
- The IAT repo uses the same Drive file and states: "We have already split the EXDark dataset with train set (80%) and test set (20%), see paper MAET (ICCV 2021)". It also gives a `wget` command for Linux. — [IAT_mmdetection README](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_high/IAT_mmdetection)
- MAET-split sizes 5,896 / 1,467 (SGLDet, AMIEOD), with these reference numbers:
  - YOLOv3-608, 25 epochs: 76.4 → SGLDet 78.6.
  - YOLOv3-640, 100 epochs: 76.5 → AMIEOD 82.1.
  - Source: the project's full-text audit, `research/literature/05_paper2_distillation_novelty_and_design.md`, citing [SGLDet ICLR 2026 pdf](https://proceedings.iclr.cc/paper_files/paper/2026/file/2b00b3331bd0f5fbfdd966ac06338f6d-Paper-Conference.pdf) and AMIEOD.
- **YOLA's ExDark and DarkFace** come from the official sources or YOLA's "customizer structure" on Google Drive.
  - The README links one Drive file ID (`1zT-Kj3nRDPp0b8SItmhB0fcZoWVdUt5m`) for both datasets.
  - Layout: `JPEGImages/`, `Annotations/*.xml` (VOC-style), and `train.txt`/`val.txt`/`test.txt`. DarkFace has `train.txt`/`val.txt` only.
  - Framework: MMDetection 3.x.
  - [YOLA repo](https://github.com/MingboHong/YOLA)
- YOLA config details:
  - ExDark YOLOv3: `train.txt`/`test.txt`, multi-scale 320–608, test 608, 24 epochs, SGD 0.001, batch 4.
  - DarkFace TOOD: train scale 750×500–1500×1000, test 1500×1000, batch 1, one class `Face`, `train.txt`/`val.txt`.
  - [yolov3_yola_exdark.py](https://github.com/MingboHong/YOLA/blob/main/configs/yolov3/yolov3_yola_exdark.py); [tood_yola_darkface.py](https://github.com/MingboHong/YOLA/blob/main/configs/tood/tood_yola_darkface.py)

**DarkFace**
- 6,000 labelled night images with face boxes, plus 9,000 unlabelled images from the same setting. — [DARK FACE page](https://flyywh.github.io/CVPRW2019LowLight/) (via search snippet; host not reachable here)
- The challenge's held-out test set is 4,000 images. The 789 images are paired low/normal-light images that are not a test set. — [UG2+ Track 2 paper, arXiv 1904.04474](https://arxiv.org/pdf/1904.04474) (via search snippet)
- Label format per secondary mirror: the first line is the face count, then one `x1 y1 x2 y2` per line. About 50,399 faces in 6,000 images. — [oschina mirror](https://git.oschina.net/CV_Lab/face-labeling/blob/master/yolov5_darkface.md) (secondary, **UNVERIFIED** against the official archive)
- Split conventions (there is no single standard):

  | Convention | Split | Reference numbers |
  |---|---|---|
  | YOLA | 5,400 / 600 | YOLOv3 60.0 → 61.5; TOOD 62.1 → 67.4 |
  | FRBNet | 3:1:1 random | YOLOv3 54.8 → 57.7; TOOD 57.0 → 65.1 |
  | SGLDet | 5,500 / 500 | — |

  Source: `research/literature/05` (full-text audit).
- FRBNet details: its 3:1:1 split is "consistent with the official experimental setup in UG2+". Training is 20 epochs at 608 (YOLOv3) or 12 epochs at 1500×1000 (TOOD). — [FRBNet arXiv 2510.23444](https://arxiv.org/pdf/2510.23444) and [code](https://github.com/Sing-Forevet/FRBNet) (via search summary)
- Licence for DarkFace: **UNVERIFIED** (site unreachable).

**NOD (Night Object Detection, BMVC 2021)**
- Dataset access goes through a Google Form (https://forms.gle/YhHYBofVjphosbeDA). The repo is MIT. Classes are person, bicycle and car. Cameras: Sony RX100 VII (3.2k images) and Nikon D750 (4.0k images). — [NOD repo](https://github.com/igor-morawski/NOD)
- The repo ships COCO-style JSONs per camera and combined, with a random 80/10/10 split. — [annotations/README.md](https://github.com/igor-morawski/NOD/blob/main/annotations/README.md)
- Computed from the JSONs:

  | Set | Images | Boxes |
  |---|---|---|
  | Combined train | 5,777 | 37,456 |
  | Combined val | 721 | 4,616 |
  | Combined test | 721 | 4,679 |
  | Sony test | 321 | 1,827 |

  Sony images are 5472×3648. Category ids: 1 person, 2 bicycle, 3 car.
- `NOD_Sony_RX100m7_test_extreme_nonextreme.json` adds a per-box `isextreme` flag. Computed: 815 extreme vs 1,012 non-extreme boxes. Boxes also carry `istruncated`. Image and annotation ids are re-enumerated per JSON, so **ids do not match across files**. — [annotations/README.md](https://github.com/igor-morawski/NOD/blob/main/annotations/README.md)

**BDD100K-night**
- 100K images, box annotations for 10 categories, image-level time-of-day tags. — [BDD100K paper, arXiv 1805.04687](https://arxiv.org/pdf/1805.04687)
- The time-of-day values are reported as `daytime`, `night`, `dawn/dusk`, `undefined`. One paper cites about 27K night images (train). — search snippets via [arXiv 2604.18256](https://arxiv.org/pdf/2604.18256); exact counts **UNVERIFIED**
- Users report label noise: images filtered as "daytime" can be night. — [bdd100k issue #20](https://github.com/bdd100k/bdd100k/issues/20)
- Official download site, label-file names (e.g. "det_20") and licence: **UNVERIFIED** (doc.bdd100k.com unreachable). A Hugging Face mirror notes that use implies agreement to the BDD100K License. — [HF mirror card](https://huggingface.co/datasets/dgural/bdd100k/raw/6d57d4a2200b3c58375747f04dd7df147cd0de1a/README.md)

**LLVIP (visible channel only)**
- 15,488 visible–infrared pairs (30,976 images), mostly very dark scenes, pedestrians labelled. — [LLVIP repo](https://github.com/bupt-ai-cz/LLVIP); [arXiv 2108.10831](https://arxiv.org/abs/2108.10831)
- Folders: `Train_LLVIP_vi`, `Test_LLVIP_vi` (and `_ir`). Annotations are VOC XML, and a converter toolbox (xml → yolo / coco) is provided. — [LLVIP README](https://github.com/bupt-ai-cz/LLVIP)
- Annotations were corrected on 2023-02-21. Download: Google Drive https://drive.google.com/file/d/1VTlT3Y7e1h-Zsne4zahjx5q0TK2ClMVv/view, or Baidu (password 14lc). — [download_dataset.md](https://github.com/bupt-ai-cz/LLVIP/blob/main/download_dataset.md)
- Boxes were annotated on the **infrared** image. Labels are therefore independent of visible-light visibility, which makes this a unique "invisible object" test. — [LLVIP paper](https://arxiv.org/html/2108.10831v4) (via snippet)
- The 12,025 / 3,463 split, 1280×1024 resolution and non-commercial licence are all **UNVERIFIED** in this session.

**COCO class indices** (Ultralytics 0-based COCO-80, for P0 mapping) — [coco.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml) (standard list; not re-fetched):
person 0, bicycle 1, car 2, motorcycle 3, bus 5, train 6, truck 7, boat 8, traffic light 9, cat 15, dog 16, bottle 39, cup 41, chair 56, dining table 60.

### Inferences

**Step-by-step dataset preparation (do once, before anything else).**

**0.1 ExDark-MAET (primary benchmark)**
1. Download the MAET/IAT archive (Drive `1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC`).
2. Use `main/train.txt` (5,896) as train and `main/val.txt` (1,467) as **test**.
3. Do not select checkpoints on it; MAET's own config does. Carve a fixed, class-stratified 10% of train (≈590 images, seed 0) as an internal val set for early-stopping sanity checks only. Better still, report the **last epoch** for every arm.
4. Convert VOC XML to:
   - COCO JSON (for MMDetection, pycocotools and TIDE);
   - YOLO txt (for Ultralytics).
5. Keep one canonical `image_id ↔ filename` table, so per-image bootstrap and lighting-condition joins work.
6. Join lighting / indoor-outdoor labels from `imageclasslist.txt` by filename.
7. Verify that the 7,363 names in MAET's lists are a subset of `imageclasslist.txt` (assert in code).

**0.2 ExDark-official (secondary)**
- Build train / val / test = 3,000 / 1,800 / 2,563 from column 5.
- Annotation source: official bbGt txt converted to COCO, or YOLA's VOC XMLs. Check that the box count per image is identical between the two.
- Report P1/P3 here so the numbers sit next to YOLA/FRBNet/Wu et al.

**0.3 Image hygiene for both ExDark splits**
- Mixed extensions and some PNGs mean you must open every image with `PIL.Image.open(p).convert('RGB')` and apply EXIF orientation consistently in every tool.
- Ultralytics and MMDetection may treat EXIF rotation differently. Normalise once by re-saving all raw images as lossless PNG with orientation applied. Test that raw-PNG P1 mAP equals raw-original P1 mAP; this is the **identity-I/O control**.

**0.4 DarkFace**
- Use YOLA's `train.txt`/`val.txt` (5,400 / 600) as primary so numbers are comparable with YOLA. Mention FRBNet's 3:1:1 and SGLDet's 5,500/500 as non-comparable conventions.
- Keep the 9,000 unlabelled images for Paper 2.
- One class: `face`.
- Tiny faces mean the image size matters. Use 1024 (long side) for Ultralytics and 1333×800 for MMDetection, identically across arms.

**0.5 NOD**
- Train on `NOD_Sony_Nikon_train.json`, select on val, test on `NOD_Sony_Nikon_test.json`.
- Additionally report Sony-test AP split by `isextreme` (815 vs 1,012 boxes). To compute it, set the non-selected boxes to `ignore`/`iscrowd=1`, so detections on them are neither true nor false positives.
- **Pre-resize** all NOD images once to long side 1333 px (area interpolation), then run every enhancer on the resized raw. Enhancers cannot run at 20 MP on 24 GB, and every arm must see identical geometry.

**0.6 BDD100K-night (nice-to-have)**
- Filter `attributes.timeofday == "night"` on train and val.
- Subsample about 10k night train images (stratified, fixed seed) to keep cost comparable. Test on all val-night images.
- Classes: the 10 detection classes for training. For P0 drop `traffic sign` (no COCO equivalent) and map pedestrian→person, rider→person (document this), car, truck, bus, train, motorcycle, bicycle, traffic light.
- Spot-check 100 "night" images for label noise.

**0.7 LLVIP-visible (nice-to-have)**
- Use `Train_LLVIP_vi`/`Test_LLVIP_vi` with the post-2023 annotations. Class: person.
- Its value: labels come from IR, so it directly tests whether enhancement recovers "invisible" objects.

**COCO mapping for P0**
- ExDark:
  - Bicycle→1, Boat→8, Bottle→39, Bus→5, Car→2, Cat→15, Chair→56, Cup→41, Dog→16, Motorbike→3, People→0, Table→60 (dining table).
  - Caveats: ExDark "Table" includes non-dining tables, and "Cup" may match COCO wine glass. Report P0 with (a) all 12 classes and (b) excluding Table and Cup as a robustness row.
- NOD: person→0, bicycle→1, car→2.
- LLVIP: person→0.
- In P0, **discard predictions of unmapped COCO classes** before evaluation; otherwise they are false positives.
- DarkFace has no COCO class. For DarkFace P0, use a detector trained on normal-light WIDER FACE (YOLOv8n and Faster R-CNN, 1 run each; WIDER FACE source **UNVERIFIED**, not checked this session). This mirrors Li et al.'s DSFD-on-DarkFace protocol.

**Darkness covariate.** Compute per-image mean luminance (Rec.601 Y on raw sRGB) and the 5th/50th percentile. Use them to (a) order datasets by darkness and (b) bin images within each dataset (quartiles), so the protocol × darkness interaction can be tested *within* a dataset as well as across datasets.

### Gaps
- DarkFace official licence, label-file format and exact archive names could not be verified (flyywh.github.io unreachable). Confirm on download.
- I could not open the BDD100K official label schema, file names or licence. Confirm `timeofday` strings and night counts after download.
- The LLVIP split sizes, resolution and licence are unverified here.
- The YOLA split files and their exact contents (5,400/600) were not opened, because the Drive archive was not downloaded. The figures rest on the project's earlier full-text audit.
- I did not open the MAET archive's `main/train.txt`/`val.txt`. The 5,896/1,467 counts come from SGLDet/AMIEOD as audited in `literature/05`. Assert the counts after download.

---

## Q2. Enhancers: repos, weights, normalisation, pitfalls, correctness checks

### Takeaway
Every enhancer is fed sRGB in [0,1] with no mean/std normalisation. The risks are elsewhere: **which tensor is saved**, **crop/pad geometry**, and **per-dataset weight choice**.
- Wrong tensor: SCI's `i` vs `r` (confirmed bug). **PairLIE's `eval.py` also writes four outputs (L, R, I, D); only `I = L^0.2·R` is the enhanced image.**
- Geometry: Zero-DCE++ crops to multiples of 12 from the top-left; Retinexformer reflect-pads to multiples of 4.
- Per-dataset weights: Retinexformer, SNR-Aware and HVI-CIDNet ship one checkpoint per training set.
- The pvnieo LIME implementation runs **DUAL by default**; `-ul` is needed for LIME.

Add **HVI-CIDNet (CVPR 2025, MIT, generalisation weights)** and **LightenDiffusion (ECCV 2024, unsupervised latent-Retinex diffusion)** as the modern arms.

### Cited Findings
- **Zero-DCE++** ([Li-Chongyi/Zero-DCE_extension](https://github.com/Li-Chongyi/Zero-DCE_extension), script [Zero-DCE++/lowlight_test.py](https://github.com/Li-Chongyi/Zero-DCE_extension/blob/main/Zero-DCE%2B%2B/lowlight_test.py)):
  - Input is `np.asarray(PIL)/255.0`, with `scale_factor = 12`.
  - The image is **cropped** `[0:h,0:w]` to multiples of 12 from the top-left.
  - Weights: `snapshots_Zero_DCE++/Epoch99.pth`. Output is saved with `torchvision.utils.save_image`.
- **SCI** ([vis-opt-group/SCI](https://github.com/vis-opt-group/SCI), CVPR 2022 code in `CVPR/`):
  - Weights: `weights/easy.pt`, `medium.pt`, `difficult.pt`. `test.py` defaults to `./weights/medium.pt`.
  - The model returns `(i, r)` with `r = clamp(input/i, 0, 1)` and `i` clamped to ≥1e-4. `test.py` saves `r`.
  - An SCI++ TPAMI 2025 version exists in `TPAMI/`.
  - [CVPR/model.py](https://github.com/vis-opt-group/SCI/blob/main/CVPR/model.py), [CVPR/test.py](https://github.com/vis-opt-group/SCI/blob/main/CVPR/test.py)
- The project's SCI results were computed from `i` (medium weights). The correct `r` has mean ≈143 on the sample and satisfies r ≥ x. — local `research/analysis/sci_bug/README.md`
- **Retinexformer** ([caiyuanhao1998/Retinexformer](https://github.com/caiyuanhao1998/Retinexformer), ICCV 2023):
  - Option files exist for LOL_v1, LOL_v2_real, LOL_v2_synthetic, SID, SMID, SDSD_indoor/outdoor, FiveK, NTIRE.
  - Pretrained models are on Google Drive/Baidu (`pretrained_weights/`).
  - The README claims support up to 4000×6000.
  - The README links its own **ExDark enhancement results** and points to IAT's mmdetection for ExDark detection.
  - `Enhancement/test_from_dataset.py` reflect-pads to multiples of `factor = 4` and has a `--GT_mean` flag ("Use the mean of GT to rectify the output") plus optional self-ensemble.
- **SNR-Aware** ([dvlab-research/SNR-Aware-Low-Light-Enhance](https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance), CVPR 2022):
  - Test configs exist per dataset (LOLv1, LOLv2_real/synthetic, SID, SMID, SDSD).
  - A single trained-models link: https://drive.google.com/file/d/1g3NKmhz7WFLCm3t9qitqJqb_J7V4nzdb/view
  - The test scripts are paired (PSNR/SSIM). The README suggests a Python 3.6 environment.
- **PairLIE** ([zhenqifu/PairLIE](https://github.com/zhenqifu/PairLIE), CVPR 2023):
  - Weights ship in-repo as `weights/PairLIE.pth`.
  - [eval.py](https://github.com/zhenqifu/PairLIE/blob/main/eval.py) computes `L, R, X = model(input)` and `I = torch.pow(L,0.2) * R  # default=0.2, LOL=0.14`, then saves **L, R, I and D** to separate subfolders.
- **KinD++** ([zhangyhuaee/KinD_plus](https://github.com/zhangyhuaee/KinD_plus), IJCV 2021): TensorFlow implementation (now TF ≥2.0). Checkpoints come from Google Drive/Baidu; inference is `python evaluate.py` on `test_images/`.
- **LIME**: no official Python code was checked. The widely used [pvnieo/Low-light-Image-Enhancement](https://github.com/pvnieo/Low-light-Image-Enhancement) implements LIME and DUAL.
  - `demo.py` defaults: `-g 0.6` gamma, `-l 0.15` lambda, `-s 3` sigma, `eps 1e-3`.
  - **LIME requires `-ul`; the default method is DUAL.** [demo.py](https://github.com/pvnieo/Low-light-Image-Enhancement/blob/master/demo.py)
- **HVI-CIDNet** ([Fediory/HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet), CVPR 2025, MIT):
  - Weights for LOLv1, LOLv2_real, LOLv2_syn (including `generalization.pth`, trained with random-gamma "to improve generalization"), plus SICE, SID, FiveK, LOL-Blur. Hosted on Hugging Face, OneDrive and Baidu.
  - Custom-folder inference: `python eval.py --unpaired --custome --custome_path <dir> --unpaired_weights <pth> [--alpha --gamma]`. Alpha default 1.0.
  - [HF paper page](https://huggingface.co/papers/2502.20272); [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_HVI_A_New_Color_Space_for_Low-light_Image_Enhancement_CVPR_2025_paper.html)
- **LightenDiffusion** ([JianghaiSCU/LightenDiffusion](https://github.com/JianghaiSCU/LightenDiffusion), ECCV 2024): unsupervised latent-Retinex diffusion. Pretrained model on Google Drive (https://drive.google.com/drive/folders/1m3t15rWw76IDDWJ0exLOe5P0uEnjk3zl) or Baidu (code cjzk). Inference: `python evaluate.py`.
- The MAET/IAT ExDark archive already contains `IMGS_Kind`, `IMGS_ZeroDCE` and `IMGS_MEBBLN` enhanced copies. — [IAT_mmdetection README](https://github.com/cuiziteng/Illumination-Adaptive-Transformer/tree/main/IAT_high/IAT_mmdetection)

### Inferences

**Enhancer protocol. Write one wrapper `enhance.py --method M --in DIR --out DIR` that does all of the following.**
1. Load with PIL `.convert('RGB')` and apply EXIF orientation once.
2. Convert to float32 in [0,1], RGB.
3. **Reflect-pad** to the method's multiple (Zero-DCE++: 12; Retinexformer: 4; others: check, and default to 8). Never crop.
4. Run under `torch.no_grad()` / `eval()` with a fixed seed.
5. Crop back to the original H×W.
6. `clamp(0,1)`, round to uint8, and save **lossless PNG** with the same basename.
7. Log per-image runtime, mean luminance, % clipped pixels and the SHA-1 of the output.
8. Assert `out.shape == in.shape`.

Labels are then reused unchanged.

**Per-method settings to freeze (record them in the paper's implementation table).**

| Method | Weights to use (primary) | Exact output to save | Settings / pitfalls | Sanity check |
|---|---|---|---|---|
| LIME | none (optimisation) | the enhanced image | pvnieo `demo.py -ul -g 0.6 -l 0.15 -s 3`. No BM3D post-denoise (state it). CPU; parallelise with `joblib` | luminance up; no NaNs; spot-check 20 images |
| Zero-DCE++ | `Epoch99.pth` | `enhanced_image` (first return) | scale_factor 12; **pad, don't crop**; convert grayscale/RGBA to RGB | output size = input; mean Y up on ≥95% of ExDark |
| SCI | `medium.pt` (ablate `difficult.pt`) | **`r`**, not `i` | input /255 only | `r ≥ x` pixel-wise on 20 images (roadmap §3); mean ≈2× raw |
| Retinexformer | `LOL_v1` (state it; ablate LOL_v2_real) | model output | reflect-pad to ×4 and crop back; **never `--GT_mean`**; no self-ensemble | reproduce LOL-v1 PSNR within ±0.3 dB of the README |
| SNR-Aware | LOLv1 checkpoint from the Drive bundle | model output | paired test script needs GT, so write an unpaired loop that rebuilds the SNR map exactly as in the repo's dataloader; padding multiple **UNVERIFIED** (check model strides) | reproduce LOLv1 PSNR |
| PairLIE | `weights/PairLIE.pth` | **`I = L^0.2 · R`** (folder `I/`), not L/R/D | keep 0.2 (default; 0.14 is the LOL setting) | mean Y up; `I/` folder used; no output equals L/R |
| KinD++ | official TF checkpoint | final restored output | TF ≥2 env; separate conda env; check whether it resizes internally (**UNVERIFIED**) | output size = input |
| HVI-CIDNet (new) | `LOLv2_syn/generalization.pth` | eval output | `--unpaired --custome`, `--alpha 1.0 --gamma 1.0` fixed; padding **UNVERIFIED** | reproduce LOLv1 PSNR with LOLv1 weights |
| LightenDiffusion (new, optional P3) | released checkpoint | eval output | stochastic: fix seed and sampler steps; may need ×64 sizes (**UNVERIFIED**); slowest | reproduce one LOL number; run twice to check determinism |

**Weight choice is a confound.** Report it explicitly, and add a one-row ablation in which each supervised enhancer is swapped between its LOL-v1 and LOL-v2-real weights (P0/P2 only, so it is cheap).

**Use the MAET-shipped `IMGS_ZeroDCE`/`IMGS_Kind` only as a cross-check.** Do not use them as arms: their settings are unknown.

**Optional Mamba arm.** E.g. Wave-Mamba or RetinexMamba, but only if the repo has released weights. Not checked here.

### Gaps
- I did not open the SNR-Aware, KinD++, HVI-CIDNet or LightenDiffusion inference code for padding/resizing behaviour, so those cells are UNVERIFIED.
- Whether the original LIME authors' code is publicly downloadable was not checked.
- No 2025–2026 Mamba LLIE with confirmed released weights was verified.

---

## Q3. Detectors: models, checkpoints, recipes, seeds, augmentation order

### Takeaway
Families:
- **One-stage CNN:** Ultralytics YOLOv8n, with s and m for the capacity sweep and YOLO11n/s as a replication.
- **DETR:** Ultralytics `rtdetr-l.pt`.
- **Two-stage:** MMDetection `faster-rcnn_r50_fpn_1x_coco`.
- **Second DETR (optional):** MMDetection `dino-4scale_r50_8xb2-12e_coco`, which replaces the expensive 50-epoch Deformable DETR. Reuse the existing raw-trained Deformable DETR only for P2.

Two Ultralytics defaults would silently break the study:
1. `optimizer=auto` switches between AdamW and MuSGD depending on iteration count. That changes the optimiser across data-scaling fractions.
2. `fraction` takes the **first k files**, which on ExDark is class-sorted.

Set the optimiser explicitly and use your own stratified subset lists.

### Cited Findings
- **Ultralytics `default.yaml` (main branch, 2026):**
  - Run settings: `epochs 100`, `patience 100`, `batch 16`, `imgsz 640`, `optimizer auto`, `seed 0`, `deterministic True`, `close_mosaic 10`, `amp True`, `fraction 1.0`, `freeze` (first N layers or a list), `lr0 0.01`, `lrf 0.01`, `warmup_epochs 3`.
  - Augmentation: `hsv_h 0.015`, `hsv_s 0.7`, `hsv_v 0.4`, `translate 0.1`, `scale 0.5`, `fliplr 0.5`, `mosaic 1.0`, `mixup 0.0`.
  - [default.yaml](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml)
- **`optimizer=auto`:** if iterations > 10,000 it uses MuSGD with lr 0.01; otherwise AdamW with `lr_fit = round(0.002*5/(4+nc), 6)`, ignoring the user's `lr0`. — [engine/trainer.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/trainer.py)
- **`fraction`:** applied as `im_files = im_files[:count]`, i.e. the first k files. — [data/base.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/base.py), [data/utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/data/utils.py)
- **`DetectionTrainer.preprocess_batch`** "move[s] a batch to the device, normalize[s] images to [0, 1], and appl[ies] optional multi-scale resizing". This is the hook where augmented batches can be enhanced online. — [models/yolo/detect/train.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/detect/train.py)
- **`init_seeds(seed, deterministic)`** seeds `random`, `numpy` and torch (all devices) and sets deterministic algorithms when requested. — [utils/torch_utils.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/torch_utils.py)
- **RT-DETR in Ultralytics:**
  - Its trainer builds `RTDETRDataset(..., augment=mode=="train", hyp=self.args)`, so the same hyperparameter/augmentation args apply. — [models/rtdetr/train.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/rtdetr/train.py)
  - `rtdetr-l.pt` gets 53.0 AP and `rtdetr-x` 54.8 AP on COCO val2017. — [RT-DETR docs](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/rtdetr.md)
- **Ultralytics checkpoints:**
  - YOLOv8n: 37.3 mAP50-95, 3.2M params. YOLOv8s: 44.9, 11.2M. — [YOLOv8 docs](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md)
  - `yolo11n.pt` … `yolo11x.pt` are available. — [YOLO11 docs](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo11.md)
- **MMDetection checkpoints:**
  - `faster-rcnn_r50_fpn_1x_coco`: 37.4 box AP. Weights: `https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth`. — [faster_rcnn/metafile.yml](https://github.com/open-mmlab/mmdetection/blob/main/configs/faster_rcnn/metafile.yml)
  - `deformable-detr_r50_16xb2-50e_coco`: 44.3 AP.
  - `deformable-detr-refine-twostage_r50_16xb2-50e_coco`: 47.0 AP. Weights: `.../deformable-detr-refine-twostage_r50_16xb2-50e_coco_20221021_184714-acc8a5ff.pth`. — [deformable_detr/metafile.yml](https://github.com/open-mmlab/mmdetection/blob/main/configs/deformable_detr/metafile.yml)
  - `dino-4scale_r50_8xb2-12e_coco`: 49.0 AP. Weights: `https://download.openmmlab.com/mmdetection/v3.0/dino/dino-4scale_r50_8xb2-12e_coco/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth`. — [dino/metafile.yml](https://github.com/open-mmlab/mmdetection/blob/main/configs/dino/metafile.yml)
- The **MMDetection COCO base train pipeline** is geometric only: `LoadImageFromFile → LoadAnnotations → Resize(1333,800, keep_ratio) → RandomFlip(0.5) → PackDetInputs`. — [_base_/datasets/coco_detection.py](https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py)
- **MMDetection runtime:** `cudnn_benchmark=False`; `DistSamplerSeedHook`. — [_base_/default_runtime.py](https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/default_runtime.py)
- **OpenMMLab 3.x seeding:** `--cfg-options randomness.seed=<s> randomness.deterministic=True`. Deterministic mode sets cudnn deterministic and disables benchmark. — [MMYOLO "set random seed"](https://mmyolo.readthedocs.io/en/latest/common_usage/set_random_seed.html)
- LIME-Eval (YOLOX-m, ExDark) found that enhance-first P3 (45.9–47.6) loses to raw (49.2), while augment-first P3 (48.5–49.2) ≈ raw. — `research/literature/06_paper1_study_novelty.md`, citing [arXiv 2410.08810](https://arxiv.org/abs/2410.08810)

### Inferences

**Frozen recipes.** These are identical across every enhancer/control arm of a given detector, with no per-arm tuning.

| Detector | Init | imgsz | Epochs | Batch | Optimiser | Aug | Notes |
|---|---|---|---|---|---|---|---|
| YOLOv8n / s / m | `yolov8{n,s,m}.pt` (COCO) | 640 (DarkFace 1024) | 100, `close_mosaic=10`, `patience=0` | 16 (m: 8) | **`optimizer=SGD, lr0=0.01, momentum=0.937`** | defaults (HSV, mosaic, flip, scale/translate) | `cos_lr=False`; save the **last** epoch |
| YOLO11n/s (replication) | `yolo11{n,s}.pt` | same | same | same | SGD as above | defaults | nice-to-have |
| RT-DETR-l | `rtdetr-l.pt` | 640 (DarkFace 1024 or 960 if OOM) | 72, `patience=0` | 8 | `optimizer=AdamW, lr0=1e-4` (fixed) | defaults | AMP on |
| Faster R-CNN R50-FPN | COCO 1x ckpt via `load_from` | (1333, 800) keep ratio | 12 (1x; steps 8/11) | 4, lr 0.005 (linear scaling from 0.02@16) | SGD | Resize+Flip (base) | `num_classes` set per dataset |
| DINO-4scale R50 (opt.) | COCO 12e ckpt | DINO default multi-scale | 12 | 2, AdamW 1e-4 ×(2/16) | AdamW | default | replaces Deformable DETR |

These epoch and lr choices are design proposals; the cited sources give only the defaults. Freeze them after a 1-seed pilot on ExDark-MAET raw. The pilot should show that YOLOv8n reaches the published Ultralytics-on-ExDark range: P1 ≈ 0.70+ mAP50 per the project's earlier P1 runs (from the doc; not re-verified).

**Seeds.**
- Ultralytics: `seed ∈ {0,1,2}`, `deterministic=True`, `workers=8`, fixed `cache=False`.
- MMDetection: `--cfg-options randomness.seed=s randomness.deterministic=True`.
- In both, log `torch.__version__`, CUDA and cuDNN, and the git hash.
- Full bit-reproducibility is not guaranteed on GPU. The seeds are for variance estimation, not exact replay.

**The two augmentation orders.**
- **Enhance-first (default):** point the dataset at the pre-enhanced PNG folder.
- **Augment-first, Ultralytics:**
  1. Subclass `DetectionTrainer`.
  2. Override `preprocess_batch(batch)`: call `super()`, which gives [0,1] on the GPU, then `batch['img'] = E(batch['img'])` under `torch.no_grad()` with the frozen enhancer in `eval()`.
  3. Do the same in a `DetectionValidator.preprocess` subclass, so val/test images are enhanced online with identical letterbox handling.
  4. Note that the gray letterbox and mosaic fill (114/255) also passes through E. That is fine, but state it.
  5. This works for the torch enhancers (Zero-DCE++, SCI, Retinexformer, PairLIE, HVI-CIDNet). LIME (CPU optimisation) and KinD++ (TF) are infeasible online; restrict augment-first to the torch "core" enhancers.
- **Augment-first, MMDetection:** subclass `DetDataPreprocessor`. Set `mean=[0,0,0]`, `std=[255,255,255]` inside, apply E to the [0,1] RGB batch, then apply ImageNet normalisation. Because the base pipeline is geometric only, the order differs mainly by resize interpolation; expect small effects for Faster R-CNN/DINO. That makes it a useful negative control.
- **Photometric-off arm (cheap, discriminative):** enhance-first plus `hsv_h=hsv_s=hsv_v=0`, run for raw and for the core enhancers. This isolates whether the order effect comes from HSV jitter acting on enhanced statistics.

**Capacity sweep:** YOLOv8 n/s/m with an identical recipe (batch 8 for m and nominal batch size `nbs=64`, so gradient accumulation equalises).

**Frozen vs fine-tuned backbone:**
- Ultralytics: `freeze=10` (YOLOv8 backbone = layers 0–9). For RT-DETR, check the layer count before using an integer.
- MMDetection: `model.backbone.frozen_stages=4`, plus `norm_eval=True`.

### Gaps
- The RT-DETR training recipe (72 epochs, AdamW 1e-4) and the Faster R-CNN lr scaling are not verified against the official docs. Confirm them in the pilot.
- I did not verify which Ultralytics augmentations RT-DETR actually applies (e.g. whether mosaic is disabled).
- Exact GPU throughput per detector on the student's GPU is unknown; measure it in step 0.6.

---

## Q4. The full run matrix (P0–P4 × enhancers × detectors × datasets × seeds), controls, scaling, interventions, analyses, priorities and GPU-hours

### Takeaway
- A must-have core (3 datasets × 3 detector families × 8 enhancers + 3 controls, 3 seeds, P0–P4, both orders for the core-4 enhancers, scaling, interventions, capacity) costs roughly **1,100–1,200 GPU-hours** on one 24 GB GPU (my estimate). That is about 7 weeks of continuous running.
- A "lean must-have" path brings it to about 700 h by giving the heavy detectors 3 seeds only on the core-4 arms.
- Training is needed only for P1, P3, P4, scaling and interventions. P0 and P2 are cheap inference.

### Cited Findings
- The roadmap requires: ≥3 detector families; ≥3 real datasets of different darkness; 3 seeds with paired-bootstrap CIs (Holm); identical tuning budgets; both augmentation orders; identity-I/O, quantization and brightness-matched gamma controls; an n/s/m capacity sweep; correlations over ≥50 cells; and a regret rate. — local `research/PUBLICATION_ROADMAP.md` §1.1
- Novel claims to support: P2 next to P3 (adaptable vs residual penalty); data scaling at 10/25/50/100% with frozen vs fine-tuned backbones; controlled interventions linked to CKA/MMD and per-object ΔAP; real-data, cross-detector size effects (opposite to LR-YOLO's synthetic trend); and ExDark's 10 lighting conditions. — local `research/literature/06_paper1_study_novelty.md` §3
- Precedent on P0 direction: on ExDark, 11 of 14 enhancers lower a frozen COCO YOLOv8 (raw 0.565). On DarkFace, all 14 raise it (raw 0.207 → 0.226–0.262). — `literature/06`, citing [CVPRW 2026 LoViF (DBDiff)](https://openaccess.thecvf.com/content/CVPR2026W/LoViF/papers/Cheng_Generative_Low-Light_Enhancement_for_Computer_Vision_Revisiting_the_Gap_Between_CVPRW_2026_paper.pdf)
- Linear CKA and minibatch CKA implementations are available as pip packages: `ckatorch` (averages HSIC over minibatches), `cka-pytorch` (`CKACalculator`, on-the-fly), `torch-cka` (AntixK), and `CKA_minibatch_pytorch`. — [ckatorch](https://pypi.org/project/ckatorch/), [cka-pytorch](https://pypi.org/project/cka-pytorch/), [torch-cka](https://pypi.org/project/torch-cka), [CKA_minibatch_pytorch](https://gh.r-mj.com/yueyang2000/CKA_minibatch_pytorch)
- TIDE: `pip install tidecv`, then `tide.evaluate(datasets.COCO(gt), datasets.COCOResult(results.json), mode=TIDE.BOX)`, followed by `summarize()` and `plot()`. — [dbolya/tide](https://github.com/dbolya/tide)

### Inferences

**Naming.**
- **Core-4 enhancers** (all torch, all usable augment-first): SCI (fixed), Zero-DCE++, Retinexformer, PairLIE.
- **Core-2** (for the most expensive experiments): SCI (zero-reference, pixel-wise gain-like) and Retinexformer (supervised transformer; denoises and alters detail).
- **Full-8:** core-4 + LIME, SNR-Aware, KinD++, HVI-CIDNet. LightenDiffusion is a 9th arm for P0/P2 only, unless time allows.
- **Controls (3):**
  - **C-id, identity I/O:** raw decoded and re-saved as PNG through the exact enhancer I/O wrapper.
  - **C-q, quantisation:** for the core-2 enhancers, store the output as 16-bit PNG and as 8-bit PNG, and compare. This measures whether 8-bit rounding of amplified shadows matters. A second variant applies the per-image gain k = mean(Y_E)/mean(Y_raw) to raw, rounds to uint8, then divides back. That measures loss from re-quantising dark codes.
  - **C-γ, brightness-matched gamma:** a per-image global gamma γ_i chosen by bisection so that mean Y(x^γ) = mean Y(E(x)). Use one C-γ per enhancer for P0/P2, and only for the core-2 in P3. This doubles as the "gain-only" intervention.

**Step 0. Infrastructure (week 1)**

| Step | What | Output |
|---|---|---|
| 0.1–0.7 | Dataset preparation from Q1 | COCO JSON + YOLO txt + `meta.csv` (image_id, file, split, lighting, in/out, meanY, p5Y) |
| 0.8 | Fix SCI and PairLIE outputs; generate all enhanced sets with `enhance.py` (Q2) | `data/<ds>/<method>/*.png` + `enh_log.csv` |
| 0.9 | `predict_dump.py`: for every (model, test set), save **COCO-format detections at conf ≥0.001, max_det 300** | `preds/<run>/<testset>.json`. All analyses read only these files. |
| 0.10 | `evaluate.py`: pycocotools mAP50 and mAP50-95, per class, per size, per image (from `COCOeval.evalImgs`) | `metrics/<run>/<testset>.parquet` |
| 0.11 | Time one epoch of each detector on each dataset, and replace the estimates below | `cost.csv` |

Use pycocotools for **all** detectors, rather than the Ultralytics or MMDetection internal mAP, so the numbers are comparable. Ultralytics' mAP uses a different interpolation. Report the Ultralytics value only in the supplement.

**Must-have runs (A-tier). Hours are my estimates for one 24 GB GPU.**

Per-run assumptions:

| Detector | Per run |
|---|---|
| YOLOv8n | ≈1.0 h (ExDark/NOD @640); ≈2.5 h (DarkFace @1024) |
| Faster R-CNN 1x | ≈2.5 h |
| RT-DETR-l | ≈4 h (@640); ≈7 h (@1024) |

| ID | Experiment | Factors | # runs | GPU-h (est.) |
|---|---|---|---|---|
| A0 | Enhancement generation | 9 methods × {ExDark-MAET test+train, DarkFace, NOD} (≈7.4k + 6k + 7.2k images) | — | ≈40 (LightenDiffusion dominates; LIME on CPU) |
| A1 | **P1** raw→raw | 3 datasets × {YOLOv8n, FRCNN, RT-DETR-l} × 3 seeds | 27 | ≈90 |
| A2 | **P0** zero-shot | COCO ckpts (YOLOv8n, FRCNN, RT-DETR-l) on ExDark and NOD × {raw, 9 enhancers, C-id, C-γ×9}; DarkFace via WIDER-FACE-trained YOLOv8n and FRCNN (2 training runs) | 2 train + ≈330 evals | ≈25 |
| A3 | **P2** raw-trained, enhanced test | the 27 A1 models × {9 enhancers, C-id, C-q, C-γ×9} | ≈540 evals | ≈15 |
| A4 | **P3 enhance-first** | YOLOv8n: (full-8 + C-id + C-q + C-γ for core-2 = 12 arms) × 3 datasets × 3 seeds = 108. FRCNN and RT-DETR-l: (core-4 + C-id) × 3 datasets × 3 seeds = 45 each | 198 | ≈500 |
| A5 | **P3 augment-first** | YOLOv8n: core-4 × 3 datasets × 3 seeds = 36. FRCNN: core-2 × {ExDark, DarkFace} × 3 seeds = 12 | 48 | ≈100 |
| A6 | **Photometric-off** | YOLOv8n {raw, core-4} × {ExDark, DarkFace} × 3 seeds | 30 | ≈45 |
| A7 | **P4 mixed** (each training image is raw or enhanced with p=0.5, resampled each epoch via online choice, or a 2× list) | YOLOv8n core-4 × 3 datasets × 3 seeds = 36; FRCNN and RT-DETR-l core-2 × 3 datasets × 3 seeds = 18 each; tested on raw and on enhanced | 72 | ≈190 |
| A8 | **Data scaling** | YOLOv8n × {ExDark, DarkFace} × {raw P1, SCI P3, Retinexformer P3, SCI P4} × {10, 25, 50%} (100% reused) × {fine-tune, frozen backbone} × 3 seeds (5 seeds at 10%), plus frozen at 100% | ≈190 | ≈90 |
| A9 | **Controlled interventions** | P0/P2 (inference): 4 interventions × full-8 × 3 datasets. P3 (training): YOLOv8n × core-2 × 4 interventions × {ExDark, DarkFace} × 3 seeds | 48 train + ≈300 evals | ≈80 |
| A10 | **Capacity sweep** | YOLOv8s, YOLOv8m × {raw, SCI, Retinexformer} P1/P3 × {ExDark, DarkFace} × 3 seeds (+P2 inference) | 36 | ≈85 |
| A11 | **Analyses** (GPU part): CKA/MMD feature extraction, per-object scores, TIDE (CPU), lighting/size (CPU) | — | — | ≈15 |
| | **Total must-have** | | ≈620 trainings | **≈1,180 GPU-h** |

**Lean path (≈700 h) if time is short.**
- A4: RT-DETR-l with 3 seeds on core-2 + C-id only, and 1 seed on the other two core enhancers.
- A7: RT-DETR-l with 1 seed.
- A10: YOLOv8s only, with YOLOv8m at 1 seed.

Do **not** cut seeds from YOLOv8n or from any ExDark/DarkFace cell that appears in the headline heatmap.

**Nice-to-have (B-tier)**

| ID | Experiment | GPU-h (est.) |
|---|---|---|
| B1 | ExDark **official split** (3000/1800/2563): YOLOv8n P1 + P3 core-4, 3 seeds (15 runs) + P2 | ≈15 |
| B2 | **BDD100K-night** (10k-image train subset): YOLOv8n P0/P1/P2/P3 core-4, 3 seeds | ≈45 + 15 enhancement |
| B3 | **LLVIP-visible**: YOLOv8n P0/P1/P2/P3 core-4, 1–3 seeds ("IR-labelled invisible objects") | ≈40 |
| B4 | **YOLO11n** replication of A1/A3/A4 on ExDark + DarkFace, core-4, 3 seeds | ≈40 |
| B5 | **DINO-4scale** P1 + P3 core-2 on ExDark + DarkFace, 1 seed; reuse the existing raw-trained Deformable DETR for P2 | ≈60 |
| B6 | LightenDiffusion P3 (YOLOv8n, 3 datasets, 3 seeds) | ≈15 + generation |
| B7 | Per-image oracle and a simple gated baseline (pick raw or E per image by a no-reference brightness rule; report regret) | CPU only |
| B8 | Faster R-CNN data scaling (ExDark only, 1 seed) | ≈40 |
| B9 | Enhancer weight-choice ablation (LOL-v1 vs LOL-v2-real) at P0/P2 | ≈3 |
| B10 | IQA (NIQE/BRISQUE/MUSIQ via `pyiqa`) vs ΔmAP correlation over ≥50 cells, with CIs (hygiene only; not a novelty claim) | CPU |

**Recommended execution order (dependencies first).**
1. Week 1: step 0. Run A0 for SCI (fixed) and PairLIE (fixed) first.
2. Week 2: A1 + A2 + A3. These give the P0/P1/P2 heatmaps early.
3. Weeks 3–5: A4.
4. Week 6: A5 + A6.
5. Week 7: A7.
6. Week 8: A8.
7. Week 9: A9 + A10.
8. Week 10: A11, statistics, figures.

Run CPU work (LIME, TIDE, bootstrap) in parallel with GPU jobs. Queue jobs with a simple `runs.csv` → `run_one.py --row k` loop, so a crash never loses the queue. Save `last.pt` and the predictions JSON, and delete intermediate checkpoints to save disk.

**Precise construction of the controlled interventions** (per image x, enhancer output y = E(x); float [0,1] RGB):
1. Luminance: Y(·) = 0.299R + 0.587G + 0.114B. Use ε = 1e-4.
2. **Gain-only (tone):**
   - x_tone = clip(x · T(Y_x)/max(Y_x, ε)), where T is the monotone tone curve that histogram-matches Y_x to Y_y (via the CDF; per image).
   - A parametric variant is C-γ (global gamma matched on mean Y).
   - x_tone keeps the raw's chromaticity, noise pattern and detail; only brightness changes.
3. **Colour-only:** x_col = clip(Y(x_tone) · (y / max(Y_y, ε))). This transplants y's chromaticity into x_tone's luminance. Equivalent in Lab: keep L* of x_tone and take a*, b* scaled from y.
4. **Detail-only and noise-only.** Use a two-band edge-preserving decomposition with a guided filter G (`cv2.ximgproc.guidedFilter`, self-guided, radius r, eps e):
   - base(z) = G_{r=8, e=1e-2}(z)
   - noise band n(z) = z − G_{r=1, e=1e-4}(z)
   - detail band d(z) = z − base(z) − n(z)
   - Construction:
     - x_detail = x_tone − d(x_tone) + d(y)
     - x_noise = x_tone − n(x_tone) + n(y)
   - Denoising enhancers therefore show up as "noise-only" (the noise band shrinks), and sharpening/structure changes as "detail-only".
   - Check the decomposition: x_tone with all three of y's components swapped in (tone, colour, d, n) must reconstruct y at PSNR ≥ 35 dB on ≥95% of images. Otherwise tune r/e, and report the residual as "interaction/other".
5. Also run **leave-one-out** variants (y with one component reverted to x_tone's), so the effects are not assumed additive. If time allows, compute a 4-factor Shapley attribution of ΔAP (16 combinations; P0/P2 inference only).

**CKA/MMD feature analysis.**
- **Models:** the raw-trained P1 models (seed 0), YOLOv8n/FRCNN/RT-DETR-l; for P3, also the matched enhanced-trained model.
- **Paired inputs:** each test image as raw and as each E/intervention.
- **Layers, YOLOv8n:** backbone indices 2, 4, 6, 9 (SPPF), and neck outputs 15, 18, 21 (the P3/P4/P5 inputs to Detect at index 22). Confirm indices by printing `model.model`.
- **Layers, Faster R-CNN:** `backbone.layer1–4`, FPN P2–P6.
- **Layers, RT-DETR-l:** the HGNetv2 stage outputs and the hybrid-encoder outputs (check names via `named_modules()`).
- Grab activations with `register_forward_hook`.
- **Representation, per image:** adaptive-average-pool to 4×4, then flatten (keeps coarse layout). **Per object:** `torchvision.ops.roi_align` 7×7 on GT boxes (at the stride of that layer), then mean-pool.
- **Metrics:**
  - linear CKA(raw feats, E feats) across images, minibatch estimator (Nguyen et al. 2021; `ckatorch` or a 20-line HSIC implementation);
  - RBF-MMD² with the median-heuristic bandwidth and an unbiased estimator, with a 1,000-permutation test;
  - per-object cosine distance.
- **Link to ΔAP:**
  - per (dataset, detector, enhancer) cell: Spearman ρ between layer-wise CKA drop and ΔmAP (P2), over ≥50 cells (8 enhancers + 4 interventions + controls, × 3 datasets × 3 detectors ≈ 120 cells);
  - per object: mixed-effects regression of per-object Δscore on the per-object feature distance, with random intercepts for image and enhancer (`statsmodels` MixedLM).

**Per-object ΔAP and regret.**
- Run `COCOeval.evaluate()` at IoU 0.5 for raw and for E with the same model.
- For each GT object o, s_M(o) = max confidence among same-class detections matched to o (0 if missed, using the greedy COCO matching).
- Δs(o) = s_E(o) − s_raw(o).
- **Object regret rate** = fraction with Δs < −0.1 (also report the threshold sweep), with a bootstrap CI.
- **Image regret rate** = fraction of images whose per-image AP50 (computed from `evalImgs`) drops.
- Stratify by COCO size (area <32², 32²–96², >96² at original resolution), by size quantiles (DarkFace is almost all "small"), by lighting condition, and by per-image darkness quartile.

**TIDE.**
- Run on P1 vs P2 vs P3 for the core-4 enhancers × 3 detectors × 3 datasets (seed 0).
- Report the dAP of Cls, Loc, Both, Dupe, Bkg and Miss.
- TIDE is hygiene; do not claim it as new.

**Lighting-condition AP (ExDark).**
- Subset `COCOeval.params.imgIds` per lighting code.
- The MAET test set is a random 20%, so expect ≈2/7 of the computed full-dataset counts. Shadow (56 images in total) and Screen (165) will be tiny: merge into 4 groups (Low+Weak+Shadow; Ambient+Twilight; Object+Single+Screen+Window; Strong) for inference, and show all 10 only descriptively.
- Also use the official test set (2,563 images), where Shadow has only 14 images, as the second source.

### Gaps
- All GPU-hour figures are estimates. They assume a 24 GB card (RTX 3090/4090/A5000 class) and have not been measured.
- Guided-filter radii, the 0.1 regret threshold and the 4×4 pooling are design proposals. Sanity-check them in a 50-image pilot.
- The A2 DarkFace P0 depends on a WIDER FACE-trained detector; that dataset's download and licence were not checked here.

---

## Q5. Statistics: bootstrap, multiple comparisons, rank tests, tools

### Takeaway
- Every comparison is **paired at the image level**: the same resampled image indices for both arms.
- Use B = 10,000 resamples for headline cells and B = 2,000 for the grid, with percentile 95% CIs.
- Seeds go in as an outer level: average the metric over seeds inside each resample, or use a two-level bootstrap.
- Apply Holm–Bonferroni within each figure or table family.
- Use Friedman + Nemenyi critical-difference diagrams via `autorank` over blocks = (dataset × detector × protocol) cells.
- Formally test the headline interaction with a mixed model ΔmAP ~ protocol × darkness.

### Cited Findings
- `autorank` (`pip install autorank`; `autorank(data, alpha=0.05)`, `plot_stats`, `create_report`, `latex_table`) automates test selection. For more than two populations that are non-normal or heteroscedastic it uses Friedman with the Nemenyi post-hoc test, computes the critical distance itself, and applies Bonferroni-style alpha adjustment for normality tests. — [sherbold/autorank](https://github.com/sherbold/autorank)
- TIDE is `pip install tidecv` and accepts COCO-format GT/results. — [dbolya/tide](https://github.com/dbolya/tide)
- Minibatch CKA estimator: Nguyen, Raghu & Kornblith (ICLR 2021), as implemented in `ckatorch`. — [ckatorch](https://pypi.org/project/ckatorch/)
- The project's audit flags LIME-Eval's statistical weaknesses: one detector, differences of about 0.4 mAP with no seeds or CIs, and a ρ computed from 15 points. — `research/literature/06` §5

### Inferences

**Bootstrap implementation (fast, exact COCO semantics).**
1. Run `COCOeval.evaluate()` once per (run, test set). Keep `evalImgs`, the per-image, per-category arrays of `dtScores`, `dtMatches`, `gtIgnore` and `dtIgnore`.
2. For each resample, draw image indices with replacement. Re-run your own `accumulate` (a copy of pycocotools' `accumulate` taking an image-multiplicity weight vector), so duplicated images count twice.
3. This avoids re-matching and makes B = 10,000 feasible on CPU, at seconds to minutes per cell.
4. Validate: with weights all 1, it must reproduce pycocotools mAP to 1e-6.

**Paired Δ.** For each resample b: Δ_b = mAP_E(b) − mAP_raw(b), using the identical index set. Report the median Δ, a percentile 95% CI, and the two-sided p = 2·min(P(Δ_b ≤ 0), P(Δ_b ≥ 0)).

**Seeds.**
- Primary: for each resample b, compute mAP for each of the 3 seeds on the same indices and average them. This is a "seed-averaged" estimand and a CI over images.
- Secondary (supplement): two-level bootstrap, resampling seeds with replacement and then images, so the CI includes seed variance.
- Also report the seed std per cell.

**Multiple comparisons.**
- Use Holm within each family. A family is, for example, "all enhancer-vs-raw tests for one (dataset, detector, protocol)": 8–12 tests.
- For the headline heatmap, also report Benjamini–Hochberg q-values in the supplement.
- Mark cells that are Holm-significant at 0.05 with a dot.

**Ranking.**
- `autorank` with treatments = {raw, 8 enhancers} and blocks = every (dataset × detector × protocol) cell (≈3 × 3 × 4 = 36 blocks).
- Produce CD diagrams per protocol (P0, P2, P3, P4) and pooled.
- Use `alpha=0.05` and `force_mode='nonparametric'` to fix Friedman + Nemenyi.

**Interaction test (headline claim 1).**
- Fit a linear mixed model with `statsmodels`: ΔmAP_cell ~ protocol × darkness + (1 | enhancer) + (1 | detector).
- Darkness can be the dataset's median Y, or the per-image darkness quartile using per-quartile ΔmAP.
- Confirm with a permutation test that shuffles darkness labels across datasets (only 3–5 levels, so state the limitation) and with the within-dataset quartile analysis (many levels).

**Decomposition CIs.** P3 − P2 and P1 − P3 are differences of paired quantities. Bootstrap them on the same indices, so the CI for each component is valid.

**Correlations.** Use Spearman ρ with a bootstrap CI over cells (≥50 cells; report n), and Fisher-z only as a check.

**Effect-size floor.** Pre-register (in the paper's text) that |Δ| < 0.5 mAP50 counts as "no practical effect", even when significant.

**Tools.**
- `pycocotools` (or `faster-coco-eval` for speed; UNVERIFIED equivalence, validate against pycocotools)
- `tidecv`, `autorank`, `scipy.stats`, `statsmodels`
- `ckatorch` (or your own HSIC code)
- `pandas`/`pyarrow` for metrics
- `matplotlib`/`seaborn` for figures

### Gaps
- I did not verify autorank's exact argument names (`force_mode`) against its current API docs; check them with `help(autorank)`.
- No source was checked for the right B for COCO-mAP bootstraps. 2,000/10,000 is a conventional choice (my inference).

---

## Q6. Every table and figure, ablations, paper outline and page budget (PR / TCSVT / TMM)

### Takeaway
- Plan **8 main figures and 6 main tables**. Everything else (per-class APs, full 10-condition plots, all seeds, Ultralytics-mAP mirrors) goes to the supplement.
- Venue guidance, from partial sources:
  - **TCSVT:** IEEE double-column with roughly 11 free pages (2025 APC list reading; 2026 figures unconfirmed); 2026 APC $2,800 for open access.
  - **TMM:** historically 8 pages before overlength charges (old source).
  - **Pattern Recognition:** single-column Elsevier; current limits not verified.
- Plan a ~12-page IEEE-style main text plus a supplement, which is portable to all three.

### Cited Findings
- TCSVT author guidelines list APC for articles submitted in 2026: $2,800 (5% IEEE-member discount). The 2025 IEEE APC list row reads "TCSVT Hybrid $2,645 $125 11 5 14", which suggests $125 per overlength page with 11 free pages for regular papers (column headers were garbled, so this is unconfirmed). An older TCSVT page said $175 per page beyond 9 (outdated). — [TCSVT Guidelines for Authors](https://ieee-cas.org/publication/tcsvt/guidelines-authors) (via search; host unreachable); [IEEE APC list 2026-07-30](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/2026-07-30-IEEE-Article-Processing-Charges-List.pdf)
- IEEE TMM, historically: "Papers submitted … in excess of 8 pages will be subject to mandatory overlength page charges." — [IEEE Content Gazette Oct 2015](https://www.nxtbook.com/nxtbooks/ieee/content_gazette_1015/index.php?startid=34) (old; current value **UNVERIFIED**)
- Pattern Recognition: the official Guide for Authors was not retrieved. A third-party guide says there is no fixed main-text word limit. — [manusights PR guide](https://manusights.com/blog/pattern-recognition-submission-guide) (secondary); journal page: [Elsevier PR](https://shop.elsevier.com/journals/pattern-recognition/0031-3203)
- Morawski et al. published a closely related zero-shot evaluation in IEEE TMM 2025, so TMM is a natural venue. — local `research/PUBLICATION_ROADMAP.md`

### Inferences

**Main figures (what each shows).**
1. **Fig. 1, teaser (½ column):**
   - Left: two example images (ExDark Ambient, DarkFace) with raw vs SCI vs Retinexformer detections.
   - Right: a 2×2 "protocol × darkness" mini-plot of ΔmAP (P0 and P3; ExDark vs DarkFace) with CIs.
   - Message: the sign depends on protocol *and* darkness.
2. **Fig. 2, protocol diagram (full width):** P0–P4 pipelines, the two augmentation orders, where controls enter, and the decomposition arrows (P3 − P2 = adaptable, P1 − P3 = residual).
3. **Fig. 3, ΔmAP heatmaps (full width; the core result):**
   - Rows: enhancers + controls. Columns: detector × dataset. One panel per protocol (P0, P2, P3, P4).
   - Diverging colormap centred at 0; Holm-significant cells dotted.
   - Datasets ordered by darkness.
4. **Fig. 4, decomposition bar chart:** per (dataset, detector), stacked bars of the adaptable vs residual penalty for the core-4, with CIs. Show enhance-first vs augment-first side by side, to show how much "residual" is an augmentation-order artefact.
5. **Fig. 5, scaling curves:** ΔmAP(E − raw) vs training fraction (10/25/50/100%, log-x), solid for fine-tuned and dashed for frozen backbone, ExDark and DarkFace, shaded 95% CIs.
6. **Fig. 6, interventions:** grouped bars of ΔmAP per intervention (tone / colour / detail / noise / full) for each enhancer at P0 and P2 (and P3 for the core-2), with leave-one-out bars or a Shapley breakdown.
7. **Fig. 7, representation shift:**
   - (a) Layer-wise CKA(raw, E) curves for the 3 detectors.
   - (b) Scatter of mean CKA drop (neck layers) vs ΔmAP across ≥50 cells, with ρ and its CI.
   - (c) Per-object feature distance vs Δscore (binned).
8. **Fig. 8, stratified effects:**
   - (a) ΔAP by object size (S/M/L and size quantiles) per detector family, with LR-YOLO's synthetic direction annotated.
   - (b) ExDark lighting groups (4 merged groups; the 10-condition version in the supplement).
   - (c) NOD extreme vs non-extreme.
   - (d) Per-image darkness quartile × protocol interaction plot.
9. **Fig. 9, CD diagrams:** Friedman/Nemenyi per protocol (P0 vs P3 at minimum). Can share a row with Fig. 8 if space is tight.
10. **Fig. 10 (or supplement), qualitative grid:** 6 images × {raw, 8 enhancers, 4 interventions}, with detections, including failure cases (noise amplification creating false positives; colour shift causing misclassification).

**Main tables.**
1. **Datasets and splits:** source, split used, #train/#test images and boxes, classes, mean/p5 luminance, licence, COCO mapping.
2. **Enhancers:** venue, type (traditional, zero-reference, supervised, unsupervised, diffusion), weights used, I/O settings, runtime per megapixel, mean ΔY. Plus the controls.
3. **Main results:** mAP50 and mAP50-95 for P1 and best/worst/median enhancer under P0/P2/P3/P4, per dataset × detector, with CIs and Holm marks. The full grid goes to the supplement.
4. **Comparability table:** our YOLOv8n/FRCNN raw numbers on ExDark-MAET and on the official split next to published raw baselines (MAET, YOLA, FRBNet, SGLDet, AMIEOD). Rows from different protocols are never mixed.
5. **Augmentation-order and control ablation:** raw, C-id, C-q, C-γ, enhance-first, augment-first and photometric-off for the core-4 (YOLOv8n, FRCNN).
6. **Capacity sweep (n/s/m) and regret:** ΔmAP and object/image regret rates per model size.
7. Supplement: TIDE error breakdown, per-class AP, all seeds, IQA-vs-ΔmAP correlations, the enhancer weight-choice ablation, and the oracle/gated selection.

**Ablations to list explicitly.**
- Augmentation order.
- Photometric augmentation on/off.
- Enhancer weights (LOL-v1 vs v2).
- SCI medium vs difficult.
- Checkpoint rule (last vs best-on-held-out-val).
- Image size (DarkFace 640 vs 1024; YOLOv8n only).
- 8- vs 16-bit storage.
- Mapping robustness for P0 (with/without Table and Cup).
- Evaluation toolkit (pycocotools vs Ultralytics mAP).

**Paper outline and page budget (IEEE double-column, ≈12 pages + references; Elsevier PR ≈30–35 single-column pages).**

| Section | Content | IEEE pages |
|---|---|---|
| Abstract | Four findings with numbers | — |
| 1 Introduction | Gap (piecewise protocols in prior work); contributions 1–5 phrased as in `literature/06` §3; Fig. 1 | 1.25 |
| 2 Related work | LLIE; low-light detection (MAET, YOLA, FRBNet, SGLDet, AMIEOD); enhancement-for-recognition studies (Wu 2409.14461, LIME-Eval, LoViF, Sun 2022, Morawski TMM 2025, underwater, haze); representation similarity (CKA, V-information) | 1.25 |
| 3 Study design | Protocols P0–P4 and decomposition (Fig. 2); datasets (Tab. 1); enhancers and controls (Tab. 2); detectors and frozen recipes; interventions (precise equations); statistics | 2.5 |
| 4 Results I: protocol × darkness | Fig. 3, Tab. 3, Fig. 9 (CD), interaction model | 1.75 |
| 5 Results II: decomposition and augmentation order | Fig. 4, Tab. 5 | 1.0 |
| 6 Results III: data scaling and capacity | Fig. 5, Tab. 6 | 1.0 |
| 7 Results IV: mechanisms | Interventions (Fig. 6), CKA/MMD (Fig. 7), per-object and regret | 1.5 |
| 8 Results V: where it helps | Size, lighting, extreme (Fig. 8); qualitative (Fig. 10) | 0.75 |
| 9 Discussion and guidelines | A practitioner decision table ("when to enhance"); limitations (3–5 darkness levels; sRGB-only; enhancers frozen) | 0.75 |
| 10 Conclusion | | 0.25 |
| References | ~60–80 | ~1.5–2 |
| Supplement | Full grids, per-class, TIDE, seeds, implementation details, code/data statement | unlimited |

**Venue choice.**
- **IEEE TMM:** closest precedent (Morawski 2025) and an evaluation-friendly audience. Confirm the current page policy on the IEEE Author Center before formatting.
- **IEEE TCSVT:** ~11 free pages (unconfirmed) fits the 12-page plan with one overlength page.
- **Pattern Recognition:** single-column, more room. Expect requests for a stronger "method" angle; the interventions plus practitioner guideline serve that role.

**Reproducibility package (reviewers expect it):**
- split files (MAET, official, YOLA-DarkFace, NOD, BDD-night list);
- `enhance.py` with exact commands and output hashes;
- all predictions JSONs;
- the bootstrap, Holm, autorank notebook;
- the frozen configs (YAML/py);
- `runs.csv`.

### Gaps
- Current (2026) page limits for TMM, TCSVT and Pattern Recognition could not be retrieved; ieee-cas.org was unreachable and the Elsevier guide was not found. Check the IEEE Author Center pages for TMM and TCSVT, and the Pattern Recognition "Guide for Authors", before formatting.
- Whether TCSVT/TMM count references toward the page limit is unverified.
