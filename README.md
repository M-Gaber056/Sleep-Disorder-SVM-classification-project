# Sleep Health Prediction Using Support Vector Machine (SVM)

## Project Overview

This project applies machine learning to sleep-health data to predict whether an individual **feels rested**, using a Support Vector Machine (SVM) classification model.

The project covers data exploration, preprocessing, categorical encoding, feature scaling, class-imbalance handling, dimensionality reduction, model selection, evaluation, and deployment through a Streamlit application.

A trained model pipeline is saved as a Joblib file so it can be loaded by the application without retraining the model every time.

## Objectives

- Explore and prepare a sleep-health dataset for machine learning.
- Identify relevant features for predicting whether an individual feels rested.
- Address class imbalance using Synthetic Minority Over-sampling Technique (SMOTE).
- Build and optimize an SVM classification pipeline.
- Evaluate model performance using classification metrics and a confusion matrix.
- Provide a Streamlit interface for interacting with the trained model.

## Repository Contents

| File | Description |
|---|---|
| `DataComputationProj.py` | Python application containing the Streamlit interface and application code. |
| `DataComputationProjNotebook.ipynb` | Jupyter Notebook documenting data exploration, preprocessing, model training, hyperparameter tuning, and evaluation. |
| `sleep_health_dataset.csv` | Dataset used for the project. |
| `svm_pipeline_model_final.joblib` | Serialized trained SVM pipeline used by the application. |

## Dataset

The dataset contains **100,000 records and 32 columns**. It includes sleep-related, lifestyle, demographic, and other health-related variables.

The notebook selects `felt_rested` as the target variable.

### Target Variable

- `felt_rested`: Binary classification target representing whether an individual feels rested.

The following columns are excluded from the model inputs in the notebook:

- `felt_rested` — the target variable.
- `sleep_disorder_risk` — an alternative target not used in this model.
- `cognitive_performance_score` — an alternative target not used in this model.
- `person_id` — identifier column.
- `country` and `occupation` — columns excluded during feature selection.

The model uses the remaining selected features to predict the target.

## Methodology

### 1. Data Exploration

The notebook explores the dataset using descriptive statistics, data-type inspection, box plots, a correlation heatmap, and class-distribution plots.

These steps help examine feature distributions, potential outliers, relationships between variables, and the balance between target classes.

### 2. Data Preprocessing

The notebook applies several preprocessing steps:

- Splits the data into training and testing sets using an 80/20 split and `random_state=42`.
- Identifies numerical and categorical features.
- Removes outliers from selected numerical training features using the interquartile range (IQR) method, excluding `shift_work` from that filtering step.
- Converts categorical variables into numerical features using one-hot encoding.
- Scales numerical features using `StandardScaler`.
- Aligns the test-set columns with the training feature columns.

### 3. Handling Class Imbalance

The notebook identifies an imbalance between the two target classes and applies **SMOTE** to the training data.

This oversampling step is intended to help the model learn from the underrepresented class. The held-out test set is not resampled.

### 4. Model Pipeline

The project uses a scikit-learn pipeline containing the following components:

1. **StandardScaler** — scales input features.
2. **VarianceThreshold** — removes features with low variance using a threshold of `0.1`.
3. **Principal Component Analysis (PCA)** — retains enough components to explain `95%` of the variance.
4. **Support Vector Classifier (SVC)** — performs the binary classification.

### 5. Hyperparameter Optimization

The notebook uses `GridSearchCV` with five-fold cross-validation to compare the following SVM settings:

- `C`: `0.1`, `1`, and `10`
- `kernel`: `rbf` and `poly`

The search optimizes classification accuracy. In the recorded notebook run, the selected parameters were:

- **Kernel:** RBF
- **C:** 10

The best cross-validation accuracy was approximately **75%**.

### 6. Model Evaluation

The model is evaluated on the held-out test set using accuracy, precision, recall, F1-score, and a confusion matrix.

The notebook's recorded confusion matrix is:

| Actual / Predicted | Class 0 | Class 1 |
|---|---:|---:|
| Class 0 | 8,689 | 3,443 |
| Class 1 | 2,462 | 5,406 |

The main results from the notebook are summarized below:

| Metric | Result |
|---|---:|
| Test-set size | 20,000 records |
| Test accuracy | Approximately 70.5% based on the confusion matrix |
| Class 0 precision | 0.78 |
| Class 0 recall | 0.72 |
| Class 0 F1-score | 0.75 |
| Class 1 precision | 0.61 |
| Class 1 recall | 0.69 |
| Class 1 F1-score | 0.65 |

These figures reflect the saved notebook outputs and are specific to that run and preprocessing workflow.

## Streamlit Application

The project includes a Streamlit application in `DataComputationProj.py`. The application provides an interactive interface for using the trained model.

The serialized model is stored in:

`svm_pipeline_model_final.joblib`

The model file allows the trained pipeline to be loaded without running the full hyperparameter search again.

## Installation and Usage

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

Replace the placeholders with your actual GitHub repository URL and folder name.

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the packages used by the project:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn streamlit joblib jupyter
```

### 3. Run the Streamlit Application

Ensure that `DataComputationProj.py` and `svm_pipeline_model_final.joblib` are in the expected locations, then run:

```bash
streamlit run DataComputationProj.py
```

Streamlit will provide a local URL where the application can be opened in a browser.

### 4. Explore the Notebook

To inspect the analysis, preprocessing steps, model training, and evaluation:

```bash
jupyter notebook DataComputationProjNotebook.ipynb
```

Make sure the dataset is available at the file path expected by the notebook.

## Tools and Technologies

- **Python** — main programming language
- **Pandas and NumPy** — data manipulation and numerical computation
- **Matplotlib and Seaborn** — exploratory data visualization
- **Scikit-learn** — preprocessing, SVM classification, PCA, pipelines, hyperparameter tuning, and evaluation
- **Imbalanced-learn** — SMOTE for handling class imbalance
- **Joblib** — saving and loading the trained model
- **Streamlit** — interactive application interface
- **Jupyter Notebook** — experimentation and analysis

## Deployed Streamlit App

https://sleep-health-prediction-svm-app.streamlit.app/

The Streamlit hosted app.
Just click this link if you don't want to run it localy!

## Limitations

- The model predicts `felt_rested`, not `sleep_disorder_risk` or `cognitive_performance_score`.
- The results depend on the dataset, feature selection, preprocessing, and train/test split.
- Performance on the held-out test set is moderate, so further experimentation may improve generalization.
- The model is an educational machine-learning project and **is not a medical diagnostic tool**.
- Reliable predictions require the application to prepare input features in the same format and order expected by the trained pipeline.

## Potential Improvements

- Compare SVM with other classification algorithms.
- Evaluate additional metrics and per-class performance.
- Investigate feature importance or other interpretability techniques.
- Validate the preprocessing and inference workflow end to end.
- Test the model on independent data to assess generalization.
- Improve the Streamlit interface and deployment workflow.

## Conclusion

This project demonstrates an end-to-end machine-learning workflow for sleep-health data, from exploratory analysis and preprocessing to SVM model optimization, evaluation, model serialization, and interactive deployment.

It provides practical experience with classification, class-imbalance handling, dimensionality reduction, cross-validation, and building a user-facing machine-learning application.
