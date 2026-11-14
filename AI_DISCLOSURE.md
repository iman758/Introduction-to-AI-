# AI Use Disclosure Statement
**Course:** AI-101 Introduction to Artificial Intelligence (Fall 2026)  
**Department:** Department of Intelligent Systems, The University of Lahore  
**Instructor:** Dr. Muhammad Awais, AI Architect  
**Student Name:** Iman Fatima  
**Roll Number:** 70177713  
**Section:** Section C  

---

### Disclosure Statement for Task A2: Concrete Compressive Strength Prediction

I used **Google DeepMind Antigravity AI Assistant** for the following specific purposes:
1. Generating initial setup commands, folder structures, and repository boilerplate.
2. Formulating the initial exploratory data analysis (EDA) plotting code (distribution histograms, correlation heatmaps, and scatter plots).
3. Structuring scikit-learn `Pipeline` objects to ensure strict data-leakage prevention (fitting scalers and polynomial features on training folds only).
4. Providing syntax templates for `GridSearchCV` hyperparameter tuning and drafting the IEEE two-column report framework.

#### Parts Generated or Substantially Assisted by AI:
- `A2_Concrete_Strength/A2_notebook.ipynb`: Cell boilerplate code for EDA plots, scikit-learn Pipeline construction, GridSearchCV parameter dictionaries, and evaluation metric logging.
- `A2_Concrete_Strength/report/source/70177713_A2_Report.tex`: Initial LaTeX structural scaffolding matching the IEEE conference template.
- `A2_Concrete_Strength/report/70177713_A2_Report.pdf`: Automated ReportLab rendering script for formatting the two-column layout.

#### Verification and Checks Performed:
- I reviewed, audited, and tested every line of Python code inside the Jupyter Notebook.
- I executed the full notebook top-to-bottom in an independent environment (`python 3.14` with `jupyter nbconvert`) to verify that all cross-validation runs, GridSearchCV searches, and test-set evaluations executed reproducibly.
- I audited the generated test metrics (MAE, RMSE, $R^2$), residual distributions, and feature importance rankings against the raw dataset to ensure zero hallucinations.
- I cross-checked that all numbers cited in the report match the notebook outputs identically.

All experiments, results, and analysis in this report and repository were produced, checked, and verified by me, and I take full responsibility for this work.
