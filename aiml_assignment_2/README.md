# Machine Learning Assignment 2 – Classifier Comparison

## a. Problem statement
The objective is to implement and compare five supervised classification algorithms on one common dataset. The models are evaluated using Accuracy, AUC, Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC). A Streamlit interface allows a user to upload labelled test data, select a saved model and view its evaluation results.

## b. Dataset description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC).
- **Source:** UCI Machine Learning Repository – https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Instances:** 569.
- **Input features:** 30 numeric features computed from digitized images of fine needle aspirate (FNA) of breast masses.
- **Target:** Binary classification. In the scikit-learn version used here, `0 = malignant` and `1 = benign`.
- **Why it meets the assignment requirement:** 569 instances (more than 500) and 30 features (more than 12).
- **Split:** 80% training and 20% test, stratified, with `random_state=42`.

The dataset is available through `sklearn.datasets.load_breast_cancer`, so `train_models.py` can reproduce the experiment without manually downloading a dataset file.

## c. GitHub Repository Link
**Add your GitHub repository URL here after creating and pushing this project.**

## d. Models used and evaluation metrics

All five models use the same train/test split. StandardScaler is applied within pipelines for Logistic Regression, KNN and SVM. Decision Tree and Gaussian Naive Bayes are trained without feature scaling. AUC is calculated using the predicted probability of class 1 (benign).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy = 0.9825, AUC = 0.9954, Precision = 0.9861, Recall = 0.9861, F1 = 0.9861, MCC = 0.9623. The linear classifier provides a strong baseline and is effective when the classes are reasonably separable. |
| Decision Tree | Accuracy = 0.9123, AUC = 0.9157, Precision = 0.9559, Recall = 0.9028, F1 = 0.9286, MCC = 0.8174. The tree captures non-linear decision rules and is easy to interpret, but a single tree can be sensitive to the training split. |
| KNN | Accuracy = 0.9561, AUC = 0.9788, Precision = 0.9589, Recall = 0.9722, F1 = 0.9655, MCC = 0.9054. KNN classifies observations using nearby training samples; feature scaling is important because it uses distance calculations. |
| Naive Bayes (Gaussian) | Accuracy = 0.9386, AUC = 0.9878, Precision = 0.9452, Recall = 0.9583, F1 = 0.9517, MCC = 0.8676. This probabilistic model is fast and assumes conditional independence and Gaussian feature distributions. |
| Support Vector Machine | Accuracy = 0.9825, AUC = 0.9950, Precision = 0.9861, Recall = 0.9861, F1 = 0.9861, MCC = 0.9623. The RBF-kernel SVM can model non-linear boundaries; scaling is important for its distance and margin calculations. |
| **Overall winner** | **Logistic Regression** has the highest F1-score on this particular held-out test split. Compare all six metrics before making a final judgement; the winner can differ depending on the metric and test split. |

## Project files
```text
aiml_assignment2/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── data/
│   └── wdbc_dataset.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metrics.json
```

## How to run locally or in BITS Virtual Lab
1. Open this project folder in the Python environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `python train_models.py` to train and save the five models and create `test_data.csv`.
4. Run `streamlit run app.py`.
5. Upload `test_data.csv`, choose a classifier and inspect the metrics, confusion matrix and classification report.

## Streamlit Community Cloud deployment
1. Push the complete project folder to a GitHub repository.
2. Open https://streamlit.io/cloud and sign in with GitHub.
3. Select **New app**, choose the repository and branch, set the main file to `app.py`, and deploy.
4. Copy the working public app URL into the final submission PDF.

The Streamlit app loads the saved model files and evaluates the uploaded labelled test CSV. It does not retrain a model using the uploaded test data.

## Reproducibility note
The reported values are generated from the fixed stratified 80:20 split and the model settings in `train_models.py`. If you change the split, random seed, preprocessing or model hyperparameters, rerun training and update the table and observations accordingly.
