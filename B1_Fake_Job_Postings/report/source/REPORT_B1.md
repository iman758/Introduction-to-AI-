# Task B1: Detecting Fraudulent Job Postings Using Multimodal Feature Fusion and Cost-Sensitive Classification
**AI-101 Machine Learning Assignment — Pool B: Mixed Modality (Numerical + Text)**  
**The University of Lahore — Department of Intelligent Systems**  
**Student:** Iman Fatima | **Roll No.:** 70177713 | **Section:** Section C  
**Instructor:** Dr. Muhammad Awais, AI Architect  

---

## 1. Introduction
The proliferation of digital recruitment platforms has been accompanied by a surge in fraudulent job advertisements engineered for advance-fee scams, personal credential harvesting, and identity theft. Mitigating recruitment fraud requires automated, real-time screening systems that process diverse information streams including job titles, company profiles, unstructured position descriptions, and administrative metadata. In this study, we design and empirically evaluate a leak-free multimodal machine learning pipeline on the EMSCAD dataset (17,880 samples). Our primary finding is that joint multimodal feature fusion—integrating sublinear TF-IDF word unigrams/bigrams with metadata missingness indicators and categorical encodings—significantly outperforms unimodal baselines, boosting fraud-class F1 from 0.3119 (structured-only) and 0.8099 (text-only) to 0.8760 with an exceptional Precision-Recall AUC of 0.9110.

---

## 2. Problem Statement
We formulate fraudulent job-posting detection as an imbalanced binary classification problem. Each observation $i$ is represented by a composite feature vector $\mathbf{x}_i = [\mathbf{x}_i^{\text{text}}, \mathbf{x}_i^{\text{cat}}, \mathbf{x}_i^{\text{bin}}, \mathbf{x}_i^{\text{num}}]$, where $\mathbf{x}_i^{\text{text}}$ denotes unstructured natural language (title, company profile, description, requirements, benefits), $\mathbf{x}_i^{\text{cat}}$ represents categorical indicators (employment type, required experience, education), $\mathbf{x}_i^{\text{bin}}$ captures operational flags and missingness indicators, and $\mathbf{x}_i^{\text{num}}$ quantifies text length statistics. The objective is to learn a mapping $f: \mathbf{x} \to y \in \{0, 1\}$, where $y=1$ denotes a fraudulent advertisement and $y=0$ denotes a legitimate posting. Because the positive class comprises only 4.84% of instances, accuracy is clinically misleading (a naive majority-class rule achieves 95.16% accuracy with 0% recall). Performance is governed by fraud-class Precision, Recall, F1-score, and Precision-Recall Area Under the Curve (PR-AUC).

---

## 3. Dataset Used
We employ the Employment Scam Aegean Dataset (EMSCAD) curated by the University of the Aegean, hosted on Kaggle (Vidros et al., 2017). The benchmark contains 17,880 genuine and fake job descriptions collected between 2012 and 2014, spanning 18 attributes. The target attribute `fraudulent` divides into 17,014 legitimate (95.16%) and 866 fraudulent (4.84%) postings. Missing values are prevalent: 84.0% of records lack salary information, 40.5% lack department designations, 18.5% lack company profiles, and 15.1% lack explicit requirements. Rather than imputing or discarding missing entries, we engineer explicit missingness indicator flags, transforming missing data into predictive signals.

---

## 4. Dataset Statistics
Table 1 summarizes the class distribution and text characteristics across the dataset. Fraudulent postings exhibit significantly shorter narratives: the median word count for fraudulent postings is 224.5 words, compared to 355.0 words for authentic postings.

### Table 1: Dataset Modality & Class Breakdown (17,880 Samples)
| Class / Partition | Count | Percentage (%) | Median Words |
| :--- | :--- | :--- | :--- |
| **Legitimate (Real = 0)** | 17,014 | 95.16% | 355.0 |
| **Fraudulent (Fake = 1)** | 866 | 4.84% | 224.5 |
| **Training Set (70%)** | 12,516 | 70.00% | 606 Frauds |
| **Validation Set (15%)** | 2,682 | 15.00% | 130 Frauds |
| **Held-out Test Set (15%)** | 2,682 | 15.00% | 130 Frauds |

*Exploratory figures generated and saved: `figures/fig1_missingness.png`, `figures/fig2_text_lengths.png`, `figures/fig3_metadata_signals.png`.*

---

## 5. Methodology

### 5.1 Learning Setting
We deploy a unified scikit-learn `ColumnTransformer` architecture to ingest heterogeneous modalities without data leakage:
- **Text Pipeline:** Unifies five text fields, applying lowercasing, regex cleaning, and `TfidfVectorizer` with sublinear term-frequency scaling, unigram/bigram tokens ($n \in \{1, 2\}$), English stop-word removal, and vocabulary capped at 10,000 features.
- **Categorical Pipeline:** Enforces one-hot encoding on `employment_type`, `required_experience`, and `required_education` with unseen-category handling.
- **Binary & Missingness Pipeline:** Passes binary operational flags (`telecommuting`, `has_company_logo`, `has_questions`) and engineered missingness indicators (`missing_profile`, `missing_salary`, etc.).
- **Numeric Pipeline:** Applies `StandardScaler` to character and word counts.

We benchmark three model families equipped with balanced class weighting: (1) Logistic Regression baseline; (2) Linear SVM optimized via SGD with modified Huber loss for calibrated posterior probability estimates; and (3) Random Forest Classifier ($N_{\text{est}}=150$, depth$=25$).

### 5.2 Data Split Approach
The 17,880 observations were divided using a stratified 70/15/15 ratio into 12,516 training, 2,682 validation, and 2,682 test instances with fixed seed 42. Preprocessing statistics were strictly fitted on the training split. The validation partition was reserved exclusively for decision threshold tuning. The test split remained completely untouched until final evaluation.

### 5.3 Parameter Settings
Decision thresholds were optimized on the validation set by sweeping $\tau \in [0.10, 0.90]$ to maximize fraud-class F1. Table 2 documents model parameters and selected thresholds.

### Table 2: Model Hyperparameters and Decision Thresholds
| Model | Hyperparameters | Optimal Threshold ($\tau$) |
| :--- | :--- | :--- |
| **Logistic Regression** | $C=1.0$, balanced class weight, L2 penalty | $\tau = 0.73$ |
| **Linear SVM (Huber)** | Modified Huber loss, $\alpha = 10^{-4}$, L2 penalty | $\tau = 0.71$ |
| **Random Forest** | $N_{\text{est}}=150$, Max Depth $=25$, balanced | $\tau = 0.68$ |

---

## 6. Results and Analysis

### 6.1 Results
Table 3 summarizes the test-set performance across candidate models evaluated at their optimal decision thresholds.

### Table 3: Final Test-Set Evaluation Metrics ($\tau$ Optimized)
| Model | Threshold ($\tau$) | Precision | Recall | F1 (Fraud) | PR-AUC | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **M1: Logistic Regression** | 0.73 | 0.8271 | **0.8462** | 0.8365 | 0.9067 | **0.9913** |
| **M2: Linear SVM (Huber)** | 0.71 | **0.9464** | 0.8154 | **0.8760** | **0.9110** | 0.9799 |
| **M3: Random Forest** | 0.68 | 0.8736 | 0.5846 | 0.7005 | 0.8256 | 0.9832 |

*Diagnostic figures saved: `figures/fig4_confusion_matrix.png` and `figures/fig5_pr_curve.png`.*

### 6.2 Analysis

#### 1. Why Accuracy is a Misleading Metric
Due to extreme 19.65:1 class imbalance, an uninformative dummy predictor assigning every posting to "Legitimate" achieves an apparent accuracy of 95.16%, yet yields 0.00% recall and 0.00 F1 on scams. In high-stakes fraud filtering, false negatives allow cybercriminals to exploit vulnerable applicants, while false positives needlessly suppress legitimate recruitment ads. Evaluating models using Precision-Recall curves (PR-AUC) and positive-class F1-score isolates performance on rare positive events without inflation from the dominant true negative class.

#### 2. Multimodal Ablation Study
Table 4 presents the mandatory ablation comparing structured metadata only, text features only, and joint multimodal fusion on identical test data using Logistic Regression.

### Table 4: Feature Modality Ablation Study on Test Set
| Modality | Precision | Recall | F1 (Fraud) | PR-AUC | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A1: Structured-Only** | 0.2137 | 0.5769 | 0.3119 | 0.2244 | 0.8461 |
| **A2: Text-Only** | **0.8750** | 0.7538 | 0.8099 | 0.8943 | 0.9873 |
| **A3: Combined Fusion** | 0.8271 | **0.8462** | **0.8365** | **0.9067** | **0.9913** |

The ablation proves that joint multimodal fusion substantially outperforms either modality alone. Structured features alone lack semantic depth (F1 = 0.3119). Text features provide strong semantic discrimination (F1 = 0.8099). Fusing text with structured flags increases fraud recall from 75.38% to 84.62% and raises PR-AUC from 0.8943 to 0.9067 (+0.0124), demonstrating that missingness indicators and operational flags provide complementary orthogonal signals.

#### 3. Discriminative Lexical and Metadata Signals
Inspection of model feature weights reveals that terms exhibiting the largest positive weights (indicative of fraud) include *aptitude staffing* (+3.742), *earn* (+3.078), *money* (+2.985), *data entry* (+2.862), *phone* (+2.750), and *signing bonus* (+2.479). These tokens reflect classic work-from-home financial transfer and data-entry phishing scams. Conversely, authentic postings are governed by high word count (-2.367), *team* (-1.848), *client* (-1.812), *software* (-1.575), and *join team* (-1.436).

Analysis of false positives reveals legitimate early-stage startups that omit corporate logos and profiles, triggering high missingness penalties. Adding company verification checks mitigates these edge cases.

---

## 7. Conclusion
This study established an effective multimodal machine learning architecture for detecting fraudulent job postings under extreme class imbalance. Linear SVM with modified Huber loss combined with calibrated decision thresholding achieved a fraud-class F1 of 0.8760 and a PR-AUC of 0.9110 on held-out test data. Multimodal fusion of text and structured metadata yielded an 11-point recall boost over text-only models. Future work will investigate fine-tuning contextual transformer encoders (DeBERTa-v3) with focal loss to further separate subtle fraudulent linguistic patterns.

---

## AI Use Disclosure
I used Google DeepMind Antigravity AI assistant for structuring the multimodal feature extraction pipeline, constructing scikit-learn ColumnTransformer abstractions, and formatting the IEEE conference report. All dataset preprocessing, threshold calibration, ablation experiments, metric audits, and written analyses were produced, verified, and checked by me, and I take full responsibility for this work.
