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
| **Pool B** | Mixed (Num + Text) | **B1** | Fraudulent Job-Posting Detection (EMSCAD) | [B1_notebook.ipynb](B1_Fake_Job_Postings/B1_notebook.ipynb) | [70177713_B1_Report.pdf](B1_Fake_Job_Postings/report/70177713_B1_Report.pdf) | In Progress |
| **Pool C** | Seq-to-Seq | **C2** | Abstractive News Summarisation (XSum) | [C2_notebook.ipynb](C2_News_Summarization/C2_notebook.ipynb) | [70177713_C2_Report.pdf](C2_News_Summarization/report/70177713_C2_Report.pdf) | Pending |

---

## Task A2: Concrete Strength Prediction Key Results

Evaluation performed on held-out 20% test partition (206 samples). All models used domain-engineered features ($w/c$ ratio and $\log(1+\text{Age})$):

| Model Family | Pipeline / Hyperparameters | Test MAE (MPa) | Test RMSE (MPa) | Test $R^2$ |
| :--- | :--- | :---: | :---: | :---: |
| **M1: Linear Regression (Baseline)** | StandardScaler $\to$ LinearRegression | 5.289 | 6.573 | 0.8323 |
| **M2: Polynomial + Ridge (Tuned)** | Poly(deg=3) $\to$ StandardScaler $\to$ Ridge($\alpha=0.1$) | 4.173 | 5.910 | 0.8645 |
| **M3: Random Forest (Tuned)** | StandardScaler $\to$ RF($N=300, d=20, l=1$) | **3.720** | 5.420 | 0.8860 |
| **M4: MLP Regressor** | Hidden $(128, 64)$, Adam, early stopping = 15 | 3.876 | **5.344** | **0.8892** |

* **Feature Ablation Impact:** Introducing $w/c$ and $\log(1+\text{Age})$ cut Linear Regression RMSE from **9.796 MPa** to **6.573 MPa** (a **32.9%** reduction).
* **Best Model:** Multi-Layer Perceptron (MLPRegressor) reached an RMSE of **5.344 MPa**, an **18.7%** improvement over the linear baseline.

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
├── B1_Fake_Job_Postings/              <- Pool B: Mixed modality task folder
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

# 3. Launch JupyterLab or run the notebook via CLI
jupyter lab
```
