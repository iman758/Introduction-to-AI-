# AI-101 Machine Learning Assignment: Training Models on Three Data Modalities
**Department of Intelligent Systems | The University of Lahore | Fall 2026**  
**Course Instructor:** Dr. Muhammad Awais, AI Architect  

| Student Details | Value |
| :--- | :--- |
| **Name** | Iman Fatima |
| **Roll Number** | 70177713 |
| **Section** | Section C |
| **GitHub Repository** | `https://github.com/<username>/AI101-ML-Assignment-70177713` |
| **Random Seed** | `42` (fixed across all splits and models) |

---

## Tasks Summary

| Pool | Modality | Task ID | Description | Notebook | Report PDF | Status |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Pool A** | Numerical | **A2** | Concrete Compressive Strength Prediction | [A2_notebook.ipynb](A2_Concrete_Strength/A2_notebook.ipynb) | [70177713_A2_Report.pdf](A2_Concrete_Strength/report/70177713_A2_Report.pdf) | **Completed** |
| **Pool B** | Mixed (Num + Text) | **B1** | Fraudulent Job-Posting Detection (EMSCAD) | [B1_notebook.ipynb](B1_Fake_Job_Postings/B1_notebook.ipynb) | [70177713_B1_Report.pdf](B1_Fake_Job_Postings/report/70177713_B1_Report.pdf) | **Completed** |
| **Pool C** | Seq-to-Seq | **C2** | Abstractive News Summarisation (XSum) | [C2_notebook.ipynb](C2_News_Summarization/C2_notebook.ipynb) | [70177713_C2_Report.pdf](C2_News_Summarization/report/70177713_C2_Report.pdf) | Pending |

---

## Key Results Summary

### Task A2: Concrete Strength Prediction (Pool A — Numerical)
Evaluation on held-out 20% test partition (206 samples). Domain features: $w/c$ ratio and $\log(1+\text{Age})$.

| Model Family | Pipeline / Hyperparameters | Test MAE (MPa) | Test RMSE (MPa) | Test $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **M1: Linear Regression (Baseline)** | StandardScaler $\to$ LinearRegression | 5.289 | 6.573 | 0.8323 |
| **M2: Polynomial + Ridge (Tuned)** | Poly(deg=3) $\to$ StandardScaler $\to$ Ridge($\alpha=0.1$) | 4.173 | 5.910 | 0.8645 |
| **M3: Random Forest (Tuned)** | StandardScaler $\to$ RF($N=300, d=20, l=1$) | **3.720** | 5.420 | 0.8860 |
| **M4: MLP Regressor** | Hidden $(128, 64)$, Adam, early stopping = 15 | 3.876 | **5.344** | **0.8892** |

* **Feature Ablation:** $w/c$ and $\log(\text{Age})$ cut Linear Regression RMSE from 9.796 to 6.573 MPa (**32.9%** reduction).
* **Best Model:** MLPRegressor reached **5.344 MPa** (**18.7%** lower than baseline).

### Task B1: Fraudulent Job-Posting Detection (Pool B — Mixed Modality)
Evaluation on held-out 15% test partition (2,682 samples; 130 frauds). Extreme class imbalance: 19.65 : 1 ratio.

| Model Family | Preprocessing & Threshold | Precision | Recall | F1 (Fraud) | PR-AUC | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **M1: Logistic Regression** | ColumnTransformer ($\tau = 0.73$) | 0.8271 | **0.8462** | 0.8365 | 0.9067 | **0.9913** |
| **M2: Linear SVM (Huber)** | ColumnTransformer ($\tau = 0.71$) | **0.9464** | 0.8154 | **0.8760** | **0.9110** | 0.9799 |
| **M3: Random Forest** | ColumnTransformer ($\tau = 0.68$) | 0.8736 | 0.5846 | 0.7005 | 0.8256 | 0.9832 |

* **Mandatory Modality Ablation (Logistic Regression on Test Set):**
  - Structured-Only: F1 = 0.3119, PR-AUC = 0.2244
  - Text-Only: F1 = 0.8099, PR-AUC = 0.8943
  - **Combined Fusion:** F1 = **0.8365**, PR-AUC = **0.9067** (Recall raised from 75.38% to 84.62%).
* **Best Model:** Linear SVM with modified Huber loss achieved a fraud-class F1 of **0.8760** and PR-AUC of **0.9110**.

---

## Repository Directory Structure

```
AI101-ML-Assignment-70177713/
├── README.md                          <- Main project overview & results summary
├── AI_DISCLOSURE.md                   <- Mandatory AI use disclosure statement
├── requirements.txt                   <- Complete Python dependencies
├── .gitignore                         <- Excludes datasets (>50MB), checkpoints, cache
├── A2_Concrete_Strength/              <- Pool A: Numerical regression task
│   ├── A2_notebook.ipynb              <- Fully executed Jupyter notebook with outputs
│   ├── figures/                       <- Generated high-resolution EDA and evaluation plots
│   │   ├── fig1_histograms.png
│   │   ├── fig2_correlation.png
│   │   ├── fig3_scatter.png
│   │   ├── fig4_pred_vs_actual.png
│   │   └── fig5_importance_cv.png
│   ├── data/
│   │   └── README.md                  <- Dataset acquisition guide & UCI repository link
│   └── report/
│       ├── 70177713_A2_Report.pdf     <- 2-page double-column IEEE report PDF
│       └── source/
│           ├── 70177713_A2_Report.tex <- LaTeX source code
│           └── REPORT_A2.md           <- Markdown report version
├── B1_Fake_Job_Postings/              <- Pool B: Mixed modality task
│   ├── B1_notebook.ipynb              <- Fully executed Jupyter notebook with outputs
│   ├── figures/                       <- Generated high-resolution diagnostic plots
│   │   ├── fig1_missingness.png
│   │   ├── fig2_text_lengths.png
│   │   ├── fig3_metadata_signals.png
│   │   ├── fig4_confusion_matrix.png
│   │   └── fig5_pr_curve.png
│   ├── data/
│   │   └── README.md                  <- EMSCAD dataset download guide & mirror
│   └── report/
│       ├── 70177713_B1_Report.pdf     <- 2-page double-column IEEE report PDF
│       └── source/
│           ├── 70177713_B1_Report.tex <- LaTeX source code
│           └── REPORT_B1.md           <- Markdown report version
└── C2_News_Summarization/             <- Pool C: Sequence-to-sequence task folder
```

---

## Environment Setup & Reproducibility

```bash
# 1. Clone the repository
git clone https://github.com/<username>/AI101-ML-Assignment-70177713.git
cd AI101-ML-Assignment-70177713

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch JupyterLab or run the notebooks via CLI
jupyter lab
```
