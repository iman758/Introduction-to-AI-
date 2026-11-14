# AI Use Disclosure Statement
**Course:** AI-101 Introduction to Artificial Intelligence (Fall 2026)  
**Department:** Department of Intelligent Systems, The University of Lahore  
**Instructor:** Dr. Muhammad Awais, AI Architect  
**Student Name:** Iman Fatima  
**Roll Number:** 70177713  
**Section:** Section C  

---

### Disclosure Statement for Task A2: Concrete Compressive Strength Prediction (Pool A)

I used **Google DeepMind Antigravity AI Assistant** for the following specific purposes:
1. Generating initial setup commands, folder structures, and repository boilerplate.
2. Formulating exploratory data analysis (EDA) plotting code (distribution histograms, correlation heatmaps, scatter plots).
3. Structuring scikit-learn `Pipeline` objects to ensure strict data-leakage prevention (fitting scalers and polynomial features on training folds only).
4. Providing syntax templates for `GridSearchCV` hyperparameter tuning and drafting the IEEE two-column report framework.

#### Parts Generated or Substantially Assisted by AI:
- `A2_Concrete_Strength/A2_notebook.ipynb`: Cell boilerplate code for EDA plots, scikit-learn Pipeline construction, GridSearchCV parameter dictionaries, and evaluation metric logging.
- `A2_Concrete_Strength/report/source/70177713_A2_Report.tex`: Initial LaTeX structural scaffolding matching the IEEE conference template.
- `A2_Concrete_Strength/report/70177713_A2_Report.pdf`: Automated ReportLab rendering script for formatting the two-column layout.

---

### Disclosure Statement for Task B1: Fraudulent Job-Posting Detection (Pool B)

I used **Google DeepMind Antigravity AI Assistant** for the following specific purposes:
1. Implementing the multimodal `ColumnTransformer` feature extraction pipeline (text TF-IDF vectorization, categorical one-hot encoding, binary/missingness indicator flags, numeric length scaling).
2. Implementing decision threshold sweep functions on the validation set to optimize fraud-class F1-scores.
3. Structuring the mandatory ablation study comparing structured-only, text-only, and combined modalities on the held-out test partition.
4. Generating diagnostic visualization code for confusion matrix heatmaps, Precision-Recall curves, and drafting the IEEE conference report text.

#### Parts Generated or Substantially Assisted by AI:
- `B1_Fake_Job_Postings/B1_notebook.ipynb`: ColumnTransformer pipeline setup, validation threshold optimizer, ablation study routines, and coefficient extraction.
- `B1_Fake_Job_Postings/report/source/70177713_B1_Report.tex`: Initial LaTeX structural scaffolding matching the IEEE conference template.
- `B1_Fake_Job_Postings/report/70177713_B1_Report.pdf`: Automated ReportLab rendering script for formatting the two-column layout.

---

### Disclosure Statement for Task C2: Abstractive News Summarisation on XSum (Pool C)

I used **Google DeepMind Antigravity AI Assistant** for the following specific purposes:
1. Fetching verified XSum sample records via the HuggingFace datasets server API for reproducible local offline execution.
2. Formulating algorithms to calculate article/summary length statistics, compression ratios, and novel unigram, bigram, and trigram percentages.
3. Constructing pure-Python reference implementations of ROUGE-1, ROUGE-2, and ROUGE-L (LCS-based) evaluation metrics.
4. Drafting sequence-to-sequence GRU attention network configurations, beam search parameters, and formatting the IEEE conference report text.

#### Parts Generated or Substantially Assisted by AI:
- `C2_News_Summarization/C2_notebook.ipynb`: Data sampling code, novel n-gram computation, ROUGE calculation functions, and qualitative case study formatting.
- `C2_News_Summarization/report/source/70177713_C2_Report.tex`: Initial LaTeX structural scaffolding matching the IEEE conference template.
- `C2_News_Summarization/report/70177713_C2_Report.pdf`: Automated ReportLab rendering script for formatting the two-column layout.

---

### Verification and Checks Performed (All Tasks):
- I reviewed, audited, and tested every line of Python code inside each Jupyter Notebook.
- I executed all notebooks top-to-bottom in an independent environment (`python 3.14` with `jupyter nbconvert`) to verify that all preprocessing, splits, training runs, threshold calibrations, and test-set evaluations executed reproducibly.
- I audited all generated test metrics (MAE, RMSE, $R^2$ for A2; Precision, Recall, F1, PR-AUC, ROC-AUC for B1; ROUGE-1, ROUGE-2, ROUGE-L for C2), confusion matrices, and ablation tables against the raw datasets to ensure zero hallucinations.
- I cross-checked that all numbers cited in the reports match the notebook outputs identically.

All experiments, results, and analysis in this report and repository were produced, checked, and verified by me, and I take full responsibility for this work.
