Breast Cancer Diagnostic Classification using Machine Learning

An end-to-end Machine Learning pipeline that analyzes diagnostic measurements and classifies breast cancer cases into malignant and benign classes.

Built with Python, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook, and Git/GitHub.

Features

• Data quality analysis — analyzes dataset dimensions, missing values, duplicate records, and target distribution
• Data preprocessing — prepares the dataset for Machine Learning using reproducible preprocessing pipelines
• Feature scaling — applies StandardScaler where required for model training
• Multiple classification models — implements Logistic Regression, Decision Tree, and Random Forest
• Model evaluation — evaluates models using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix
• Cross-validation — uses 5-fold Stratified Cross-Validation to evaluate model stability
• Hyperparameter optimization — uses Grid Search for Logistic Regression and Randomized Search for Random Forest
• Final test evaluation — evaluates optimized models on an unseen 20% test set
• Experiment tracking — stores baseline, cross-validation, optimization, and final evaluation results as CSV files
• Six-week documentation — includes individual reports covering the complete internship project

How it works

1. The Breast Cancer Wisconsin Diagnostic dataset is loaded using Scikit-learn.
2. The dataset is analyzed for dimensions, missing values, duplicates, and target distribution.
3. The data is split into training and test sets using an 80/20 stratified split.
4. Preprocessing and feature scaling are applied through Scikit-learn pipelines.
5. Three baseline classification models are trained:
   - Logistic Regression
   - Decision Tree
   - Random Forest
6. The baseline models are evaluated using multiple classification metrics.
7. 5-fold Stratified Cross-Validation is performed using ROC-AUC as the primary validation metric.
8. Logistic Regression is optimized using GridSearchCV.
9. Random Forest is optimized using RandomizedSearchCV.
10. The optimized models are evaluated on the untouched test set.
11. The final results are stored in the results/ directory.

Machine Learning Workflow

Breast Cancer Dataset
        │
        ▼
Data Loading & Understanding
        │
        ▼
Data Quality Analysis
        │
        ▼
Preprocessing & Feature Scaling
        │
        ▼
80/20 Stratified Train-Test Split
        │
        ▼
Baseline Model Training
   ┌────┼─────────────┐
   ▼    ▼             ▼
  LR   Decision      Random
       Tree          Forest
   └────┼─────────────┘
        ▼
Model Evaluation
        │
        ▼
5-Fold Stratified Cross-Validation
        │
        ▼
Hyperparameter Optimization
   ┌────┴─────────────┐
   ▼                  ▼
Grid Search      Randomized Search
   │                  │
   └────────┬─────────┘
            ▼
Final Test Evaluation
            │
            ▼
       Results Analysis

Dataset

The project uses the Breast Cancer Wisconsin Diagnostic dataset available through scikit-learn.

Dataset characteristics

• Total samples: 569
• Total features: 30
• Training samples: 455
• Test samples: 114
• Target classes: 2
• Train-test split: 80/20

The dataset contains numerical diagnostic measurements including:

• Radius
• Texture
• Perimeter
• Area
• Smoothness
• Compactness
• Concavity
• Concave points
• Symmetry
• Fractal dimension

The measurements are provided across mean, standard error, and worst-case feature groups.

Models implemented

Logistic Regression

A linear classification model implemented with feature standardization through a Scikit-learn pipeline.

Decision Tree

A tree-based classification model used as a non-linear baseline.

Random Forest

An ensemble classification model consisting of multiple decision trees.

Model evaluation

The following metrics are used:

• Accuracy
• Precision
• Recall
• F1-score
• ROC-AUC
• Confusion Matrix

5-fold Stratified Cross-Validation is used to evaluate model stability while maintaining the class distribution across validation folds.

Hyperparameter optimization

Two optimization strategies were implemented.

Logistic Regression

Optimization method: Grid Search

Best configuration:

C = 1
solver = liblinear
class_weight = None

Best cross-validation ROC-AUC:

0.9960

Random Forest

Optimization method: Randomized Search

Best configuration:

n_estimators = 400
max_depth = 10
min_samples_split = 2
min_samples_leaf = 4
max_features = sqrt
class_weight = None

Best cross-validation ROC-AUC:

0.9901

Final results

The optimized models were evaluated on the unseen test set.

Tuned Logistic Regression

Accuracy  : 0.9825
Precision : 0.9861
Recall    : 0.9861
F1-score  : 0.9861
ROC-AUC   : 0.9957

Confusion Matrix:

[[41  1]
 [ 1 71]]

Tuned Random Forest

Accuracy  : 0.9561
Precision : 0.9589
Recall    : 0.9722
F1-score  : 0.9655
ROC-AUC   : 0.9927

Confusion Matrix:

[[39  3]
 [ 2 70]]

Final comparison

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Tuned Logistic Regression | 98.25% | 98.61% | 98.61% | 98.61% | 99.57% |
| Tuned Random Forest | 95.61% | 95.89% | 97.22% | 96.55% | 99.27% |

Based on the final test evaluation, Tuned Logistic Regression achieved the strongest overall performance among the optimized models.

Project structure

breast-cancer-ml-classification/
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── notebooks/
│   ├── Week_2_Data_Preprocessing.ipynb
│   ├── Week_3_Model_Implementation.ipynb
│   ├── Week_4_Model_Evaluation.ipynb
│   └── Week_5_Model_Optimization.ipynb
├── reports/
│   ├── Week_1_Project_Planning.docx
│   ├── Week_2_Preprocessing_Feature_Engineering.docx
│   ├── Week_3_Model_Implementation.docx
│   ├── Week_4_Model_Evaluation_Validation.docx
│   ├── Week_5_Model_Optimization_Experimentation.docx
│   └── Week_6_Final_Comprehensive_Report.docx
├── results/
│   ├── baseline_results.csv
│   ├── cross_validation_results.csv
│   ├── optimization_results.csv
│   └── optimized_test_results.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── optimize.py
│   └── final_evaluation.py
├── .gitignore
├── requirements.txt
└── README.md

Setup

1. Clone the repository

git clone https://github.com/Madhankumars-ece/breast-cancer-ml-classification.git
cd breast-cancer-ml-classification

2. Install dependencies

python -m pip install -r requirements.txt

3. Run data loading

python src/data_loader.py

4. Run preprocessing

python src/preprocessing.py

5. Train baseline models

python src/train.py

6. Evaluate baseline models

python src/evaluate.py

7. Run hyperparameter optimization

python src/optimize.py

8. Run final evaluation

python src/final_evaluation.py

Jupyter Notebooks

The repository includes interactive notebooks for the major implementation stages.

Week 2 — Data Preprocessing

Week_2_Data_Preprocessing.ipynb

Covers dataset loading, data quality analysis, preprocessing, feature scaling, and train-test preparation.

Week 3 — Model Implementation

Week_3_Model_Implementation.ipynb

Covers implementation and training of the baseline classification models.

Week 4 — Model Evaluation

Week_4_Model_Evaluation.ipynb

Covers model evaluation, confusion matrices, classification metrics, and cross-validation.

Week 5 — Model Optimization

Week_5_Model_Optimization.ipynb

Covers Grid Search, Randomized Search, hyperparameter experimentation, and optimized model selection.

Weekly reports

The complete six-week internship documentation is available in the reports/ directory.

• Week 1 — Project Planning
• Week 2 — Preprocessing & Feature Engineering
• Week 3 — Model Implementation
• Week 4 — Model Evaluation & Validation
• Week 5 — Model Optimization & Experimentation
• Week 6 — Final Comprehensive Report

Results and experiment files

The results/ directory contains:

results/
├── baseline_results.csv
├── cross_validation_results.csv
├── optimization_results.csv
└── optimized_test_results.csv

These files contain the recorded outputs from baseline evaluation, cross-validation, hyperparameter optimization, and final optimized-model testing.

Reproducibility

The project uses reproducible experimentation practices including:

• random_state=42
• Stratified train-test splitting
• 5-fold Stratified Cross-Validation
• Scikit-learn pipelines
• Defined hyperparameter search spaces
• Separate test-set evaluation
• Structured CSV result files

Notes for talking about this in interviews

• Why multiple models?
  Logistic Regression, Decision Tree, and Random Forest provide different modeling approaches and allow meaningful baseline comparison.

• Why ROC-AUC?
  ROC-AUC provides a threshold-independent measure of how effectively a classifier separates the two target classes.

• Why Stratified Cross-Validation?
  Stratification helps maintain the class distribution across validation folds.

• Why keep the test set untouched?
  The test set is reserved for final evaluation so that the reported performance represents evaluation on previously unseen data.

• Why hyperparameter optimization?
  Optimization provides a systematic way to search for model configurations that improve validation performance rather than relying only on default parameters.

• Why Logistic Regression as the final model?
  The tuned Logistic Regression model achieved the strongest overall final test performance among the optimized models, with 98.25% accuracy and 99.57% ROC-AUC.

Possible extensions

• Feature selection and dimensionality reduction
• Additional Machine Learning algorithms
• Model calibration
• Explainable AI techniques
• External dataset validation
• API-based model deployment
• Web-based prediction interface
• Model monitoring and MLOps integration

Disclaimer

This project is developed for educational and Machine Learning engineering purposes.

It is not intended for clinical diagnosis or direct medical decision-making. Real-world medical applications require clinical validation, domain expertise, regulatory approval, and appropriate safety controls.

Project status

Completed — Six-Week Machine Learning Engineer Internship Project

Planning
   ↓
Preprocessing
   ↓
Model Implementation
   ↓
Evaluation & Validation
   ↓
Optimization & Experimentation
   ↓
Final Analysis
