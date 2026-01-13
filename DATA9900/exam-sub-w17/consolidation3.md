DM9900 — Deduplicated Study Q&A (from consolidation2.txt)

## Data Mining Foundations

### 1) What is **Data Mining** and how is it different from **Machine Learning**?
**Answer:**  
- **Data Mining**: the end-to-end process of discovering patterns/insights from data (includes cleaning, transforming, modeling, interpretation).  
- **Machine Learning (ML)**: a set of algorithms/techniques (often used within data mining) to learn patterns from data to make predictions/decisions.

### 2) What are the key advantages of ML and what is meant by **Big Data**?
**Answer:**  
- **ML advantages**: automation of decisions, scalability to large datasets, improved accuracy from learned patterns, adaptability as more data arrives, cost reduction, pattern recognition beyond human limits.  
- **Big Data**: datasets too large/fast/diverse for traditional tools; often described by **Volume, Velocity, Variety**.

### 3) Give common real-world applications of data mining.
**Answer:**  
Healthcare (risk/diagnosis), finance (fraud/credit), retail (recommendations/market basket), marketing (churn/targeting), manufacturing (predictive maintenance), telecoms (churn/network), social media (sentiment/trends), transport (routing/demand forecasting).

---

## Data Mining Life Cycles (KDD & CRISP-DM)

### 4) What is **KDD** and what are its steps?
**Answer:**  
**KDD (Knowledge Discovery in Databases)** is an iterative process to find valid/novel/useful/understandable patterns.  
Typical steps:
1. **Selection** (choose relevant data)  
2. **Preprocessing** (clean/integrate)  
3. **Transformation** (format/feature creation)  
4. **Data Mining** (apply algorithms)  
5. **Interpretation/Evaluation** (validate, make sense of results)  
6. **Knowledge representation** (communicate findings)

### 5) What are the phases of **CRISP-DM**?
**Answer:**  
1. Business Understanding  
2. Data Understanding  
3. Data Preparation  
4. Modeling  
5. Evaluation  
6. Deployment

---

## Data Preprocessing

### 6) What is **data preprocessing** and why is it important?
**Answer:**  
Preparing raw data for analysis (cleaning, transforming, organizing). It improves data quality and often improves model accuracy, stability, and efficiency.

### 7) What are common ways to handle **missing values**?
**Answer:**  
- **Deletion**: remove rows (if few missing) or columns (if too many missing).  
- **Imputation**: mean/median (numeric), mode (categorical), forward/backward fill (time series), **k-NN imputation**, regression-based imputation, hot deck (borrow from similar records).

### 8) What is **data discretization** and how can it be done?
**Answer:**  
Turning continuous values into bins/categories. Methods include:
- equal-width binning  
- equal-frequency binning  
- clustering-based binning  
- entropy/information-gain-based binning

### 9) What is **data integration** and what issues can arise?
**Answer:**  
Combining data from multiple sources. Issues: schema conflicts, type mismatches, redundancy, integrity constraints. Benefit: more complete and potentially higher-quality dataset.

### 10) What is **data transformation**? Give common examples.
**Answer:**  
Changing data representation/scale, e.g. normalization, standardization (z-score), log transforms, encoding categorical variables (one-hot), aggregation.

---

## Feature Selection & Dimensionality Reduction

### 11) What is the difference between **feature selection** and **feature extraction**?
**Answer:**  
- **Selection**: choose a subset of existing features.  
- **Extraction**: create new transformed features (e.g., PCA components).

### 12) What is **PCA** and when should you use it?
**Answer:**  
**Principal Component Analysis** creates orthogonal components that capture maximum variance.  
Use it for high-dimensional data to reduce complexity, speed up modeling, reduce overfitting risk, and aid visualization (at the cost of interpretability).

### 13) What are **filter** vs **wrapper** feature selection methods?
**Answer:**  
- **Filter**: selects features using statistics (e.g., correlation); fast; model-independent.  
- **Wrapper**: searches subsets using model performance; slower; can overfit but may improve results for a specific model.

---

## Clustering

### 14) What is clustering and what is it used for?
**Answer:**  
Unsupervised grouping of similar data points. Used for segmentation, structure discovery, anomaly detection, and simplifying data.

### 15) How does **K-means** work and how do you pick **K**?
**Answer:**  
Iterative: initialize centroids → assign points to nearest centroid → update centroids → repeat until stable.  
Pick **K** using **elbow method**, **silhouette score**, gap statistic, and domain knowledge.

### 16) What is **hierarchical clustering** and what is a **dendrogram**?
**Answer:**  
Hierarchical clustering builds a merge/split hierarchy of clusters; a **dendrogram** visualizes merge distances. You choose cluster count by “cutting” the dendrogram at a height.

### 17) What are common hierarchical linkage methods?
**Answer:**  
- **Single** (closest points)  
- **Complete** (farthest points)  
- **Average** (average pairwise distance)  
- **Ward** (minimizes within-cluster variance)

---

## Time Series

### 18) What is a time series and what patterns can it contain?
**Answer:**  
Ordered observations over time. Patterns: **trend**, **seasonality**, **cyclicity**, and **noise**.

### 19) Compare **Euclidean distance**, **DTW**, and **SAX** for time series similarity.
**Answer:**  
- **Euclidean**: fast; assumes alignment; poor with phase shifts.  
- **DTW (Dynamic Time Warping)**: non-linear alignment; handles time shifts/speed changes; more expensive.  
- **SAX**: symbolic compression (normalize → PAA → discretize); good for indexing/search/clustering at scale.

### 20) Why do we use **windowing** in time series feature extraction?
**Answer:**  
Windows capture local behavior; sliding windows track evolving patterns. Window size trades off detail vs stability and compute cost.

### 21) Name common feature types extracted from time series.
**Answer:**  
Statistical (mean/variance/min/max), trend (slope), frequency (FFT components), temporal (autocorrelation), symbolic (SAX).

### 22) What is **ARIMA** used for?
**Answer:**  
A classic forecasting approach combining autoregression + differencing (to enforce stationarity) + moving average.

---

## Text Mining & NLP

### 23) What is **NLP**, and what makes it difficult?
**Answer:**  
NLP deals with understanding/generating human language. Difficulties: ambiguity, context dependence, variability (dialects/styles), sarcasm/irony.

### 24) What is the difference between **Bag of Words**, **TF‑IDF**, and **word embeddings**?
**Answer:**  
- **BoW**: counts words; sparse; ignores order/semantics.  
- **TF‑IDF**: BoW weighted by rarity across corpus; stronger for retrieval/classification.  
- **Embeddings (Word2Vec/GloVe/FastText)**: dense vectors capturing semantic similarity; useful for deeper NLP tasks.

### 25) What is **tokenization**, **stop-word removal**, **stemming**, and **lemmatization**?
**Answer:**  
- **Tokenization**: split text into units (words/subwords/sentences).  
- **Stop words**: remove frequent low-signal terms (may be domain-specific).  
- **Stemming**: crude suffix stripping (fast, may create non-words).  
- **Lemmatization**: dictionary/POS-based base form (more accurate, slower).

### 26) What is **POS tagging** and why is it useful?
**Answer:**  
Assigns grammatical roles (noun/verb/etc.) to tokens; supports syntactic/semantic analysis.

### 27) What issues arise in supervised learning on text?
**Answer:**  
High dimensionality and sparsity, class imbalance, ambiguity/context, representation choices, noisy text (typos/slang), and labeled data scarcity.

### 28) What are the steps in a **sentiment analysis pipeline**?
**Answer:**  
Collect labeled text → preprocess → extract features → choose model → train/evaluate → interpret results → deploy/monitor/retrain.

---

## Explainability (XAI)

### 29) Explain **interpretability** vs **explainability** and the trade-off with performance.
**Answer:**  
- **Interpretability**: model is inherently understandable (e.g., linear model, small tree).  
- **Explainability**: post-hoc methods explain predictions of complex models.  
Trade-off: complex models may perform better but are harder to understand.

### 30) What is **local** vs **global** explainability?
**Answer:**  
- **Local**: explains one prediction (instance-level).  
- **Global**: explains overall model behavior across the dataset.

### 31) What are **LIME** and **SHAP**?
**Answer:**  
- **LIME**: fits a simple surrogate model around a single point using perturbed samples (local explanation).  
- **SHAP**: uses Shapley-value ideas to attribute feature contributions (local + can aggregate to global); often more principled but computationally heavier.

### 32) What are **model-agnostic** vs **model-specific** explainers?
**Answer:**  
- **Model-agnostic**: works with any black box (e.g., LIME/SHAP).  
- **Model-specific**: uses internal structure (e.g., linear coefficients, tree feature importance).

### 33) What are **pre-model**, **in-model**, and **post-model** explainability strategies?
**Answer:**  
- **Pre-model**: engineer/select interpretable features; reduce bias in data.  
- **In-model**: use interpretable models or built-in interpretability constraints.  
- **Post-model**: explain after training (LIME/SHAP/counterfactuals/visualizations).

### 34) Define: **accountability**, **trustworthiness**, **confidence**, **causality**, **fairness**.
**Answer:**  
- **Accountability**: traceability/auditability of decisions and responsibility.  
- **Trustworthiness**: reliability and consistent performance.  
- **Confidence**: stated certainty of a prediction (often probability-like).  
- **Causality**: cause-effect reasoning beyond correlation.  
- **Fairness**: avoiding discriminatory outcomes across groups; ethically aligned decisions.

---

## Evaluation Metrics & Validation

### 35) What are TP, TN, FP, FN in a confusion matrix?
**Answer:**  
- **TP**: predicted positive, actually positive  
- **TN**: predicted negative, actually negative  
- **FP**: predicted positive, actually negative  
- **FN**: predicted negative, actually positive

### 36) Define **accuracy**, **precision**, **recall**, **F1**.
**Answer:**  
- Accuracy = (TP+TN)/(TP+TN+FP+FN)  
- Precision = TP/(TP+FP)  
- Recall = TP/(TP+FN)  
- F1 = 2·(Precision·Recall)/(Precision+Recall)

### 37) What are **ROC** and **AUC** used for?
**Answer:**  
ROC plots TPR (recall) vs FPR across thresholds; **AUC** summarizes ROC area (threshold-independent separability measure).

### 38) What is cross-validation and name common types.
**Answer:**  
Repeated train/test splitting to estimate generalization. Types: **k-fold**, **stratified k-fold**, **leave-one-out**, **time-series CV** (respects ordering).

### 39) Define **MAE**, **MSE**, **RMSE**, and **R²** for regression.
**Answer:**  
- MAE: average absolute error (robust-ish to outliers).  
- MSE: average squared error (penalizes large errors).  
- RMSE: sqrt(MSE), same units as target.  
- R²: proportion of variance explained.

---

## Core Supervised Models (High-Level)

### 40) Linear vs Logistic Regression: what do they predict?
**Answer:**  
- **Linear regression**: continuous values.  
- **Logistic regression**: class probability (typically binary) via logistic function.

### 41) What is **KNN** and what are its strengths/weaknesses?
**Answer:**  
Predicts using majority (classification) among the **k** nearest points.  
Strengths: simple, flexible boundaries. Weaknesses: slow at prediction for big datasets, sensitive to scaling/irrelevant features.

### 42) What is an **SVM** and what does the kernel do?
**Answer:**  
Finds a hyperplane maximizing margin between classes; kernels enable non-linear boundaries by implicit feature-space mapping (e.g., RBF).

### 43) What is **Naïve Bayes** and what is the “naïve” assumption?
**Answer:**  
Bayesian probabilistic classifier assuming **feature independence** given the class; often strong for text despite assumption violations.

### 44) Decision Trees vs Random Forests: key difference?
**Answer:**  
A **Decision Tree** is a single interpretable model prone to overfitting; **Random Forest** averages many trees (bagging + feature randomness) for better generalization but reduced interpretability.

### 45) What are Neural Networks / Deep Learning good for, and why are they “recently” dominant?
**Answer:**  
Good for complex, high-dimensional data (images, text, sequences). Recent dominance comes from better hardware, large datasets, and improved architectures/optimizers.

---

## Association Rules (Apriori)

### 46) What is an association rule and what are antecedent/consequent?
**Answer:**  
Rule form: **Antecedent → Consequent** (If LHS items occur, RHS items likely occur).  
Antecedent = condition; consequent = predicted co-occurrence.

### 47) Define **support**, **confidence**, **lift**.
**Answer:**  
- **Support**: frequency of itemset in transactions.  
- **Confidence(A→B)**: P(B|A).  
- **Lift(A→B)**: how much more often A and B occur together than if independent (lift > 1 indicates positive association).

### 48) What does **Apriori** do at a high level?
**Answer:**  
Generates candidate itemsets, counts support, prunes infrequent sets, then derives rules filtered by confidence/lift thresholds.

---

## Reinforcement Learning (RL)

### 49) What is RL and how does it relate to an **MDP**?
**Answer:**  
RL learns by interaction to maximize cumulative reward. An **MDP** formalizes RL with states, actions, transitions, rewards, and the Markov property.

### 50) What are the **Bellman equation** and **Temporal Difference (TD)** learning?
**Answer:**  
- **Bellman equation**: recursive relationship defining value functions.  
- **TD learning**: updates value estimates using the error between predicted and observed + bootstrapped next-state value.

### 51) What is the **exploration vs exploitation** trade-off? Name a common strategy.
**Answer:**  
Choosing between trying new actions (explore) vs using known best actions (exploit). Common strategy: **ε-greedy**.

### 52) What is **Deep Q-Learning** and what stabilizes it?
**Answer:**  
Uses a neural net to approximate Q-values. Stability techniques: **experience replay** and a **target network**.