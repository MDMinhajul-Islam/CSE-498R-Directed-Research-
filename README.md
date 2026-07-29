# Instance-Aware Object Removal & High-Fidelity Background Reconstruction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8x-seg](https://img.shields.io/badge/Detector-YOLOv8x--seg-00FFFF?logo=ultralytics&logoColor=black)
![LaMa](https://img.shields.io/badge/Global%20Inpaint-LaMa-6E44FF)
![Stable Diffusion 2](https://img.shields.io/badge/Local%20Refine-Stable%20Diffusion%202-10B981)
![Colab](https://img.shields.io/badge/Runs%20on-Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)

> **Full title:** *Instance-Aware Object Removal and Background Reconstruction Using Hybrid Inpainting with Comprehensive Evaluation*

Removing people from crowded, real-world photos and rebuilding the hidden background so the edit looks natural is a genuinely hard problem. Classical inpainting smears textures and leaves halos; diffusion inpainting produces plausible textures but breaks down when a large region is masked. **PhantomFill** addresses this with a structured, multi-stage pipeline that detects and segments every person, refines the removal mask, reconstructs the background with a **hybrid global (LaMa) + local (Stable Diffusion 2)** strategy, and — crucially — introduces a **residual-aware evaluation** that checks whether the removed people are still detectable in the output.

This work was developed for **CSE 498R: Directed Research (Spring 2026)** at North South University under the supervision of **Dr. Jilan Samiuddin**.


![Mask Refinement](assets/mask_refinement.png)


![Baseline Comparison](assets/baseline_comparison.png)


## Repository Structure

```text
CSE-498R-Directed-Research/
├── assets/                             # Figures used in this README
├── 498R Proposal Form.pdf              # Research proposal
├── Work plan.pdf                       # Weekly work plan and methodology
├── update1/                            # Early masking & removal tests
├── update2/                            # Batch experiments
├── update3/                            # Diffusion inpainting baseline
├── Update_5_cse_498R.ipynb            # Consolidated pipeline update
├── Week_4_AZ.zip                       # Week-4 results (figures, metrics.csv, grids)
└── WEEK-6/
    ├── statistical_analysis_week6.ipynb        # Mask coverage vs quality analysis
    └── week-6 updates/
        ├── cse498r_final_stage0_colab.ipynb    # Dataset → auto mask generation
        ├── cse498r_final_stage1_colab.ipynb    # Selective segmentation + inpainting
        ├── cse498r_final_stage2_advanced_colab.ipynb
        ├── cse498r_final_stage2_improved_v2.ipynb   # Final proposed pipeline
        └── stage3_V1(learning_based_local_refinement_model).ipynb
```

The `stage2_improved_v2` notebook is the finalized pipeline used for evaluation.

---

## Getting Started

The notebooks run in **Google Colab** (GPU runtime recommended; T4 or A100).

1. Open `WEEK-6/week-6 updates/cse498r_final_stage2_improved_v2.ipynb` in Google Colab.
2. Enable a GPU runtime: *Runtime → Change runtime type → GPU*.
3. Run the setup cell to install dependencies:
   ```bash
   pip install ultralytics simple-lama-inpainting diffusers transformers \
               lpips opencv-python-headless Pillow matplotlib pandas tqdm scikit-image
   ```
4. Upload a dataset ZIP of person images (MS COCO subset), or mount Google Drive.
5. Run the cells in order: dataset extraction → mask generation → refinement → LaMa → Stable Diffusion refinement → residual-aware evaluation. Metrics are written to CSV for analysis.

---

## Methods & Tools

| Component             | Tool / Model                    |
| --------------------- | ------------------------------- |
| Detection & segmentation | YOLOv8x-seg (Ultralytics)    |
| Global inpainting     | LaMa (`simple-lama-inpainting`) |
| Local refinement      | Stable Diffusion 2 (Diffusers)  |
| Mask processing       | OpenCV (morphology, feathering) |
| Metrics               | PSNR, SSIM, OUT-SSIM, LPIPS + residual detection |
| Dataset               | MS COCO (multi-person subset)   |
| Environment           | Google Colab / Kaggle (PyTorch, CUDA GPU) |

---

## Limitations & Future Work

* No clean ground-truth backgrounds exist for real COCO scenes, so evaluation relies on proxy and residual-based metrics.
* Complete semantic removal in dense, heavily overlapping crowds remains unsolved (avg. ~6.96 residual detections).
* Diffusion refinement is computationally heavy (~7.9 s/image) — not yet real-time.
* **Future:** synthetic paired ground-truth dataset, detector-guided residual refinement loop, YOLO+SAM ensemble masks, and ControlNet-guided diffusion to reduce hallucination.

---

## Team & Academic Context

* **Course:** CSE 498R — Directed Research (Spring 2026)
* **Institution:** North South University, Department of Electrical and Computer Engineering
* **Supervisor:** Dr. Jilan Samiuddin (Assistant Professor, ECE)

| Team Member          | Student ID |
| -------------------- | ---------- |
| Md. Minhajul Islam   | 2211022042 |
| Azmain Iqtidar Arnob | 2211786042 |
| Aqib Ahmed           | 2211335042 |
| Atikul Islam Nahid   | 2211978042 |
