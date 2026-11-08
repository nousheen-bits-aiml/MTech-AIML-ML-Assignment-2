# Wisconsin Diagnostic Breast Cancer — Classifier Studio

**M.Tech (AIML) | Machine Learning | Assignment 2**

This project implements and compares five supervised machine-learning classifiers on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. It also provides an interactive Streamlit application where a user can upload labelled test data, select a trained classifier, and inspect its evaluation results.

Github Repository link: https://github.com/nousheen-bits-aiml/MTech-AIML-ML-Assignment-2/tree/main/aiml_assignment_2
Live Stremlit App link: https://wisconsin-daignostic-breastcancer-classifier.streamlit.app

## 1. Problem Statement

The objective is to build and evaluate five classification models on the same binary-classification dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Each model is evaluated using the six metrics specified in the assignment: **Accuracy, AUC, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC)**.

The Streamlit application loads the previously trained model selected by the user. It evaluates that model on the uploaded labelled test CSV; it does **not** retrain a model on the uploaded test data.

## 2. Dataset Description

- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Public source:** [UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Instances:** 569
- **Input features:** 30 numerical features
- **Task:** Binary classification
- **Target labels used by the scikit-learn dataset:** `0 = malignant`, `1 = benign`

The dataset exceeds the assignment minimum requirements of 500 instances and 12 features.

### Train/test split

The data is divided into **80% training and 20% testing**, using a stratified split with `random_state=42`. Stratification preserves the class proportions in the two subsets. The reported results correspond to this fixed split and the model settings in `train_models.py`.

## 3. GitHub Repository Link

[Open the GitHub project folder](https://github.com/nousheen-bits-aiml/MTech-AIML-ML-Assignment-2/tree/main/aiml_assignment_2)

The project folder contains the Streamlit application, model-training script, dependency list, test data, dataset file, and saved model artifacts.

### Project structure

```text
aiml_assignment_2/
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

The `model/` directory stores the five trained classifiers and the recorded evaluation metrics. `test_data.csv` is the labelled held-out test set used by the Streamlit app.

## 4. Models Used and Evaluation Metrics

All five classifiers were trained and evaluated using the same dataset and fixed train/test split. The table below reports all six evaluation metrics required by the assignment for every classifier.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | **0.9825** | **0.9954** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| K-Nearest Neighbors (KNN) | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine (SVM) | **0.9825** | 0.9950 | **0.9861** | **0.9861** | **0.9861** | **0.9623** |

### Observations on Model Performance

| ML Model Name | Observation about Model Performance |
| --- | --- |
| **Logistic Regression** | It performed strongly as a linear baseline, achieving 0.9825 accuracy, 0.9954 AUC and 0.9861 F1-score. It has the highest AUC among the five models in this experiment. |
| **Decision Tree** | It can represent non-linear decision rules and is relatively easy to interpret. It achieved 0.9123 accuracy and 0.9286 F1-score. It had the lowest accuracy and MCC in this comparison. |
| **K-Nearest Neighbors (KNN)** | It classifies observations using nearby training samples, so feature scaling is important. It achieved 0.9561 accuracy and 0.9655 F1-score, giving a strong result but below Logistic Regression and SVM on this split. |
| **Gaussian Naive Bayes** | This is a fast probabilistic classifier based on conditional independence and Gaussian feature assumptions. It achieved 0.9386 accuracy and 0.9517 F1-score. Its AUC was 0.9878, despite its lower accuracy than KNN, Logistic Regression and SVM. |
| **Support Vector Machine (SVM)** | The RBF-kernel SVM can learn non-linear decision boundaries. It achieved 0.9825 accuracy, 0.9950 AUC and 0.9861 F1-score, matching Logistic Regression on accuracy and F1-score in this test. |

### Overall Winner

**Logistic Regression is the best overall model for this experiment.** Logistic Regression and SVM tie for the highest accuracy (0.9825), precision (0.9861), recall (0.9861), F1-score (0.9861), and MCC (0.9623). Logistic Regression has a slightly higher AUC (0.9954 compared with 0.9950 for SVM), so it is selected as the overall winner based on the reported results. This conclusion applies to the specified dataset split and model settings.

## 5. Streamlit Application

**Live application:** [Open Classifier Studio on Streamlit Community Cloud](https://wisconsin-daignostic-breastcancer-classifier.streamlit.app)

The application provides the following assignment features:

- **CSV upload:** Upload the labelled `test_data.csv` file containing all 30 input features and the `target` column.
- **Classifier selection:** Select one of the five saved classifiers using the model-selection dropdown.
- **Evaluation metrics:** View Accuracy, AUC, Precision, Recall, F1-score and MCC for the selected model.
- **Detailed evaluation:** View the confusion matrix and classification report.
- **Model loading:** The app loads the saved model from `model/` and evaluates it on the uploaded test data. It does not train or retrain models on the uploaded test file.

## 6. Running the Application Locally or in BITS Virtual Lab

Open a terminal in the `aiml_assignment_2` project directory and install the dependencies:

```bash
pip install -r requirements.txt
```

To train the five models and generate the saved model files and test CSV:

```bash
python train_models.py
```

To start the Streamlit application:

```bash
streamlit run app.py
```

When the app opens, upload `test_data.csv` and select a classifier to inspect its results.

## 7. Reproducibility and Notes

- The comparison results are from the stratified 80:20 train/test split with `random_state=42`.
- The same held-out test data is used to compare all five models.
- The Streamlit application uses the saved model artifacts; uploading the test CSV does not trigger model training.
- If the split, random seed, preprocessing, or model settings are changed, rerun `train_models.py` and update the comparison table and observations with the new results.
- The BITS Virtual Lab execution screenshot is a separate item in the final assignment PDF; it is not a substitute for the source code or README in the GitHub repository.
