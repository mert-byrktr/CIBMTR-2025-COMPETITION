This repository contains the solution designed for predicting transplant survival outcomes in patients undergoing allogeneic Hematopoietic Cell Transplantation (HCT) competition. The main objective is to develop models that are accurate in ranking patient risk and equitable across different racial groups.

My approach leverages a wide variety of target transformations, several families of machine learning and deep learning models, and an ensemble optimizer to combine their predictions for the best stratified performance.

**Competition Overview**

**Task**:
Predict event-free survival among patients undergoing HCT. The data includes censored survival times and an event indicator.

**Data Description**:
The provided dataset contains 59 features covering demographic and clinical factors for both recipients and donors. The target is encoded via two variables:

`efs`: The event indicator (1 for event occurrence, 0 for censoring).

`efs_time`: The observed time until event or censoring.

**Evaluation Metric**:
To ensure equitable performance across different races, predictions are evaluated using a modified stratified concordance index (C-index). This metric is computed for each race group and then combined as:

`Score = Mean(C-index of race groups) – Standard Deviation(C-index of race groups)`
A higher score indicates both high overall ranking accuracy and minimal performance variance across racial groups.

## Data Preprocessing
The preprocessing module is responsible for:

**Loading Data**:
The DataPreprocessor class takes the paths for training and test CSV files and reads them into Pandas DataFrames.

**Identifying Features**:
Designated a list of columns to remove (e.g., "ID", "efs", "efs_time", and target "y"). All other columns are treated as features. Within these features, the code identifies categorical variables by checking if the data type is "object" and stores the remaining as numerical features.

**Handling Missing Values and Encoding**:
Categorical features are filled with a placeholder (e.g., "NAN") and then factorized (i.e., label encoded) so that each unique category is mapped to an integer. Numerical features are converted to lower precision (float32 or int32) to improve memory usage.


**Final Output**:
The preprocessor returns the processed feature sets (for train and test) as well as the target information (stored separately in a DataFrame containing "ID", "efs", "efs_time", and "race_group"). This target information is used later by the custom scoring function.

## Models and Transformations
Our repository includes the implementation and training of various models, each using different transformations to better capture survival data properties:

**Gradient Boosting Models**:
Models built with libraries like `XGBoost`, `CatBoost`, and `LightGBM` are trained using various target transformations such as quantile transformations, time-weighted risk functions, and survival-specific objectives (e.g., Cox and AFT losses). These transformations help in handling the censoring in time-to-event data.

Below is an excerpt you can add to the README under the "Models and Transformations" section (or within a dedicated subsection for gradient boosting models) that specifically details the target transformations used for gradient boosting methods:

---

### Gradient Boosting Models: Target Transformations

For the gradient boosting models (using XGBoost and CatBoost), experimented with a variety of target transformations along with classical methods such as KaplanMeier, NelsonAalen, BreslowFleming etc. to better model survival outcomes and handle the challenges of censoring in time-to-event data. Below are the key transformations applied:

1. **Quantile Transformation (`transform_time_qcut`):**  
   - **Purpose:** Discretizes continuous survival times into ordinal quantile bins.  
   - **Method:** The function divides the event times (from patients who experienced the event) into `q` quantile bins using ordinal encoding; censored observations are assigned the highest ordinal value. This standardizes the target scale, enabling the model to learn from both eventful and censored cases.

2. **Time Weighted Risk (`transform_time_weighted_risk`):**  
   - **Purpose:** Incorporates time-based weighting into the risk estimate.  
   - **Method:** For patients with the event, the risk is modeled with an exponential decay relative to the observed time, while for censored observations, a linear scaling is applied. This transformation captures the intuition that earlier events should be prioritized, while still accounting for long-term survivors.

3. **Negative Log Survival (`transform_neg_log_survival`):**  
   - **Purpose:** Emphasizes differences in survival probabilities across patients.  
   - **Method:** By fitting a Kaplan–Meier estimator to the survival data, survival probabilities are computed at observed times. The negative logarithm of these probabilities is then taken, which compresses high survival probabilities and stretches low ones to enhance the model's discriminatory power.

4. **Martingale Residuals (`transform_martingale_residuals`):**  
   - **Purpose:** Provide a residual measure from a baseline Cox model that emphasizes deviations from expected survival.  
   - **Method:** Using the Nelson–Aalen estimator to compute the cumulative hazard, martingale residuals are calculated as the difference between the observed event indicator and the estimated cumulative hazard. This transformation highlights cases where the observed data deviates significantly from baseline expectations.

5. **Partial Hazard Transformation (`transform_partial_hazard`):**  
   - **Purpose:** Offers an alternative risk metric based on the hazard function.  
   - **Method:** A Cox proportional hazards model is fitted to the data, and the resulting partial hazard predictions are used as the target. Although not used in every experiment, it was explored as another way to capture relative risk.

6. **Time Buckets (`transform_time_buckets`):**  
   - **Purpose:** Approximates the risk function in a piecewise manner.  
   - **Method:** Event times are segmented into buckets based on quantiles; different constant weights are assigned to each bucket. This method provides a simplified, yet informative, discretization of survival risk over time.


**Neural Networks**:
Two neural network approaches are implemented:

A `PyTorch Lightning–based model (LitNN)` is designed with a custom multi-layer perceptron backbone, optionally with an auxiliary task to improve ranking performance. This model directly optimizes a pairwise ranking loss that compares subject pairs.

A `TensorFlow/Keras MLP baseline` uses categorical embeddings (via Keras Embedding layers) along with numerical input features.

**TabM and Pairwise Ranking Models**:
Our more advanced models employ a TabM architecture with efficient ensemble adapters and a custom pairwise ranking loss. These models compute losses over all valid patient pairs (using techniques like hinge loss with margins) to directly optimize for a ranking measure that aligns with the survival outcome discrimination.

Each of these models outputs out-of-fold predictions saved as pickle files. These predictions are later converted to rank order (using SciPy’s rankdata) before being combined in the ensemble.

### Cross-Validation Strategy
To ensure robust evaluation and to minimize overfitting, our gradient boosting models were trained and validated using the following cross-validation strategy:

**K-Fold & Stratified K-Fold**:

In experiments where maintaining the distribution of the key categorical variable is crucial (e.g., race group or the combination of race and the event indicator), I applied StratifiedKFold. This ensures that each fold retains a similar distribution of the target or stratification variable, which is especially important for equitable performance across diverse patient groups.

**Stratification Variables**:

When using stratified CV, I commonly stratify by the race group, or by a composite stratification (e.g., a combination of race and the event indicator), so that the folds reliably represent the diversity in the population.

## Ensembling Technique
The ensembling is performed in two phases:

**Rank-Based Aggregation**:
Since the different models might be on different scales, each model’s out-of-fold predictions are converted into their rank order. This normalization by ranking makes the ensemble more robust to scaling differences across models.

**Optuna-Based Weight/Selection Optimization**:
In one approach, a simple Optuna objective function suggests integer weights (for three ensemble predictions) in a specified range (e.g., 1 to 10). The ensemble prediction is computed as a weighted sum (via a dot product) of these ranked predictions.

In a more advanced implementation (encapsulated in the EnsembleOptimizer class), predictions are loaded from a diverse pool of models, including base models with fixed weights and additional models where inclusion is decided via Optuna's categorical suggestion (0 to drop, 1 to include). The objective function sums the rank-transformed predictions (multiplied by their corresponding weight or inclusion indicator) and then uses our custom scoring function (which computes the stratified C-index adjusted for variability across race groups) as the metric to maximize.

