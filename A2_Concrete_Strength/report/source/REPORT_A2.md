# Task A2: Predicting Concrete Compressive Strength Using Non-Linear and Ensemble Machine Learning Models
**AI-101 Machine Learning Assignment — Pool A: Numerical Modality**  
**The University of Lahore — Department of Intelligent Systems**  
**Instructor:** Dr. Muhammad Awais, AI Architect  
**Student:** [Iman Fatima] | **Roll No.:** 70177713 | **Section:** [Section C]  
**GitHub Repository:** [Repository Link]

---

## 1. Introduction
Testing standard concrete specimens in civil engineering requires curing durations of up to 28 days before compressive failure testing can certify structural compliance. This latency hinders rapid optimization of novel eco-friendly binders, such as blast furnace slag and fly ash replacements. Accurate data-driven prediction models enable rapid mix proportion evaluation, material cost reduction, and enhanced job-site safety. In this work, we implement an end-to-end machine learning pipeline spanning rigorous exploratory data analysis (EDA), domain-guided feature engineering, leakage-free cross-validation, and systematic model comparison. Our main finding is that non-linear models (MLP and tuned Random Forest) achieve a test-set RMSE of `[5.344]` MPa ($R^2 = `[0.889]`$), outperforming the baseline linear model by `[18.7]%` and resolving the severe under-fitting caused by non-linear hydration kinetics.

---

## 2. Problem Statement
We formulate concrete compressive strength prediction as a supervised tabular regression task. Given an input feature vector $\mathbf{x} \in \mathbb{R}^d$ representing mix components (cement, blast-furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate in $\text{kg/m}^3$) and curing age in days, the objective is to learn a hypothesis $f: \mathbb{R}^d \to \mathbb{R}$ predicting the continuous compressive strength $y \in \mathbb{R}^+$ in megapascals (MPa). Model performance is evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Coefficient of Determination ($R^2$). The core learning challenges comprise the pronounced non-linearity between curing age and strength, feature zero-inflation in supplementary cementitious materials, and the strict requirement to prevent data leakage during scaling and preprocessing.

---

## 3. Dataset Used
The dataset utilized is the Concrete Compressive Strength benchmark from the UCI Machine Learning Repository (Dataset ID 165), curated by I-Cheng Yeh (1998). The dataset comprises 1,030 experimental mixture observations and 8 real-valued input features without missing attributes. The inputs quantify the mass of cementitious materials, chemical admixtures, aggregate fractions, and curing age (ranging from 1 to 365 days). The continuous target variable is the measured 28-day or test-age compressive strength (2.33 to 82.60 MPa, $\mu = 35.82$ MPa, $\sigma = 16.71$ MPa). Preprocessing consisted of checking for missing and abnormal values (0 missing entries found) and standardizing all input features using z-score normalization ($\mu = 0, \sigma = 1$) fitted strictly on the training partition.

---

## 4. Dataset Statistics
Table 1 summarizes the descriptive statistics of the input features and target variable. Age exhibits pronounced positive skewness (3.27), with the majority of tests performed at or before 28 days and sparse testing up to 365 days. Chemical admixtures (fly ash, slag, superplasticizer) feature significant zero-inflation, reflecting mixes where these additives were omitted.

### Table 1: Dataset Descriptive Statistics (1,030 Samples)
| Feature / Target | Mean | Std | Min | Q1 | Median | Q3 | Max | Skew | Missing |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cement** ($\text{kg/m}^3$) | 281.17 | 104.51 | 102.00 | 192.38 | 272.90 | 350.00 | 540.00 | 0.51 | 0 |
| **Blast Furnace Slag** ($\text{kg/m}^3$) | 73.90 | 86.28 | 0.00 | 0.00 | 22.00 | 142.95 | 359.40 | 0.80 | 0 |
| **Fly Ash** ($\text{kg/m}^3$) | 54.19 | 63.99 | 0.00 | 0.00 | 0.00 | 118.30 | 200.10 | 0.54 | 0 |
| **Water** ($\text{kg/m}^3$) | 181.57 | 21.35 | 121.75 | 164.90 | 185.00 | 192.00 | 247.00 | 0.07 | 0 |
| **Superplasticizer** ($\text{kg/m}^3$) | 6.20 | 5.97 | 0.00 | 0.00 | 6.40 | 10.20 | 32.20 | 0.91 | 0 |
| **Coarse Aggregate** ($\text{kg/m}^3$) | 972.92 | 77.75 | 801.00 | 932.00 | 968.00 | 1029.40 | 1145.00 | -0.04 | 0 |
| **Fine Aggregate** ($\text{kg/m}^3$) | 773.58 | 80.18 | 594.00 | 730.95 | 779.50 | 824.00 | 992.60 | -0.25 | 0 |
| **Age** (days) | 45.66 | 63.17 | 1.00 | 7.00 | 28.00 | 56.00 | 365.00 | 3.27 | 0 |
| **Strength (Target)** (MPa) | 35.82 | 16.71 | 2.33 | 23.71 | 34.45 | 46.13 | 82.60 | 0.42 | 0 |

*Exploratory figures generated and saved: `figures/fig1_histograms.png`, `figures/fig2_correlation.png`, `figures/fig3_scatter.png`.*

---

## 5. Methodology

### 5.1 Learning Setting
We frame the problem as supervised regression across four candidate model families:
1. **Linear Regression (Baseline):** Provides an interpretable, parametric reference to quantify performance limits under linear assumptions.
2. **Polynomial + Ridge Regression:** Incorporates quadratic and interaction terms ($\binom{10+2}{2} = 66$ features) coupled with $L_2$ Tikhonov regularization ($\alpha$) to capture non-linear relationships while curbing multicollinearity.
3. **Random Forest Regressor:** A non-parametric ensemble of decorrelated decision trees that naturally accommodates non-linear step boundaries and high-order feature interactions without requiring linear assumptions.
4. **Multi-Layer Perceptron (MLP):** A feedforward artificial neural network with two hidden layers (128 and 64 neurons) and ReLU activations, trained with Adam and early stopping on validation loss.

Domain-guided feature engineering creates two new features: the water-to-cement ratio ($w/c = \text{Water}/\text{Cement}$) and logarithmic age ($\log(1+\text{Age})$), expanding the feature space to 10 dimensions.

### 5.2 Data Split Approach
The dataset of 1,030 samples was partitioned into 80% training (824 samples) and 20% test (206 samples) sets using a fixed random seed (42). To strictly prevent data leakage, all preprocessing transformations (StandardScaler, PolynomialFeatures) were encapsulated within scikit-learn `Pipeline` objects. Transformation parameters were fitted exclusively on the training split and applied to the test split during evaluation. Hyperparameter tuning was conducted using 5-fold cross-validation strictly on the 824 training samples.

### 5.3 Parameter Settings
Hyperparameter spaces were explored via GridSearchCV on the training set. Table 2 documents the search grids and chosen configurations.

### Table 2: Model Hyperparameter Search Space and Selected Values
| Model | Search Space | Selected Configuration |
| :--- | :--- | :--- |
| **Linear Regression** | None (Ordinary Least Squares) | Default OLS |
| **Poly + Ridge** | Degree $\in \{1, 2, 3\}$, $\alpha \in \{0.01, 0.1, 1, 10, 100\}$ | Degree = `[3]`, $\alpha = `[0.1]`$ |
| **Random Forest** | $N_{\text{est}} \in \{100, 200, 300\}$, Max Depth $\in \{\text{None}, 10, 20\}$, Min Leaf $\in \{1, 2, 4\}$ | $N_{\text{est}} = `[300]`$, Depth = `[20]`, Min Leaf = `[1]` |
| **MLP Regressor** | Hidden: $(128, 64)$, lr: $10^{-3}$, Adam, early stopping = 15 | Layers: $(128, 64)$, lr: $10^{-3}$, Adam |

---

## 6. Results and Analysis

### 6.1 Results
Table 3 reports the held-out test set performance across all four candidate models evaluated on identical test samples.

### Table 3: Final Test-Set Evaluation Metrics (Bold indicates best)
| Model | MAE (MPa) | RMSE (MPa) | $R^2$ |
| :--- | :--- | :--- | :--- |
| **M1: Linear Regression (Baseline)** | `[5.289]` | `[6.573]` | `[0.8323]` |
| **M2: Poly + Ridge (Tuned)** | `[4.173]` | `[5.910]` | `[0.8645]` |
| **M3: Random Forest (Tuned)** | **`[3.720]`** | `[5.420]` | `[0.8860]` |
| **M4: MLP Regressor** | `[3.876]` | **`[5.344]`** | **`[0.8892]`** |

*Evaluation figures generated and saved: `figures/fig4_pred_vs_actual.png` (Predicted vs Actual & Residual plot) and `figures/fig5_importance_cv.png` (Feature Importance & CV RMSE bar chart).*

### 6.2 Analysis

#### 1. Linear Baseline Under-performance and Violated Assumptions
The standard linear model under-performs because concrete strength development violates the core assumption of linearity: $\mathbb{E}[y|\mathbf{x}] = \mathbf{x}^T \boldsymbol{\beta}$. Chemical hydration kinetics dictated by Abram's law follow a power/logarithmic curve over time. As demonstrated in our EDA, strength increases rapidly within the first 14 days and asymptotes thereafter. Linear regression enforces a constant marginal strength gain per curing day, severely over-predicting strength at intermediate ages and under-predicting mature samples. Furthermore, the residuals exhibit heteroscedasticity across the strength spectrum.

#### 2. Feature Engineering Ablation Study
Table 4 details the impact of introducing the water-to-cement ratio ($w/c$) and $\log(1+\text{Age})$.

### Table 4: Feature Engineering Ablation on Test Set
| Configuration | MAE (MPa) | RMSE (MPa) | $R^2$ |
| :--- | :--- | :--- | :--- |
| **Linear Reg (8 base features)** | `[7.746]` | `[9.796]` | `[0.6276]` |
| **Linear Reg (10 engineered features)** | `[5.289]` | `[6.573]` | `[0.8323]` |
| **Random Forest (8 base features)** | `[3.791]` | `[5.532]` | `[0.8812]` |
| **Random Forest (10 engineered features)** | `[3.722]` | `[5.397]` | `[0.8869]` |

The engineered features produce a dramatic improvement for Linear Regression, decreasing RMSE by `[32.9]%` (from `[9.796]` to `[6.573]` MPa) and boosting $R^2$ from `[0.6276]` to `[0.8323]` (+`[0.2047]`). For the Random Forest, the gain is modest (RMSE decreases from `[5.532]` to `[5.397]` MPa, a 2.4% reduction). This differential occurs because decision trees recursively partition raw feature space with orthogonal splits, natively synthesizing step-wise approximations of non-linear ratios and logarithmic curves. Conversely, linear models lack interaction capability unless explicitly furnished with multiplicative and non-linear terms.

#### 3. Error Distribution and Practical Engineering Insights
Inspection of model residuals demonstrates that the largest absolute prediction errors ($>10$ MPa, occurring in `[18]` out of 206 test samples) occur primarily in early-age specimens ($\le 7$ days curing) and ultra-high-strength mixes ($>70$ MPa). For samples with $\text{Age} \le 7$ days, the RMSE is `[3.898]` MPa, driven by rapid cement hydration dynamics sensitive to unmeasured laboratory variables such as relative humidity and ambient curing temperature.

Random forest feature importance reveals that Cement (`[0.327]`), Age / $\log(\text{Age})$ (`[0.312]`), and Water/Cement ratio (`[0.141]`) collectively account for over 75% of total split impurity reduction. For practicing engineers, this confirms that water-to-cement ratio and curing duration are the primary levers governing structural capacity, while blast furnace slag and fly ash act as secondary modifiers whose contribution is realized primarily at mature curing stages ($>56$ days).

---

## 7. Conclusion
In this assignment, non-linear machine learning architectures were trained and evaluated to predict concrete compressive strength from mixture design and curing age. Multi-layer perceptrons and tuned random forests proved vastly superior to linear baselines, achieving a final test RMSE of `[5.344]` MPa and explaining `[88.92]%` of strength variance. Domain-guided feature engineering ($w/c$ ratio and logarithmic age) proved essential for parametric models, cutting linear regression error by over 32%. Future investigations could explore gradient boosting ensembles (e.g., LightGBM/CatBoost) and Physics-Informed Neural Networks (PINNs) that enforce physical hydration bounds directly within the loss function.

---

## AI Use Disclosure
I used Google DeepMind Antigravity AI assistant for structuring the exploratory data analysis pipeline, implementing scikit-learn Pipeline abstractions, and generating initial LaTeX report boilerplate. All experiments, cross-validation runs, metrics calculations, error analyses, and written interpretations were executed, verified, and checked by me, and I take full responsibility for this work.

---

## References
1. I.-C. Yeh, "Modeling of strength of high-performance concrete using artificial neural networks," *Cement and Concrete Research*, vol. 28, no. 12, pp. 1797–1808, 1998.
2. D. Dua and C. Graff, "UCI Machine Learning Repository: Concrete Compressive Strength Data Set," Irvine, CA: University of California, School of Information and Computer Science, 2017.
3. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

