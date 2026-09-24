# Blood Donor Classification

## 📌 Project Overview

This project focuses on predicting whether a person is likely to donate blood based on historical donor information. Machine learning techniques are used to preprocess the dataset, handle class imbalance, and build a classification model using the **K-Nearest Neighbors (KNN)** algorithm.

The project compares different approaches for handling imbalanced data, including:

* Random OverSampling
* SMOTE (Synthetic Minority Over-sampling Technique)

The effect of preprocessing, scaling, and outlier handling on model performance is also investigated.

---

## 📂 Dataset

The dataset contains information about blood donors and their previous donation history.

Typical features may include:

* Recency — months since the last donation
* Frequency — number of previous donations
* Monetary — total blood donated
* Time — months since the first donation
* Target — whether the person donated blood

> Adjust the feature names according to the actual dataset being used.

---

## 🔄 Data Preprocessing

The following preprocessing steps are applied before training the model.

### 1. Handling Missing Values

Missing values are identified using:

```python
df.isnull().sum()
```

Depending on the dataset, missing values can be handled by:

* Removing rows with missing values
* Replacing numerical values with the mean or median
* Using another suitable imputation method

---

### 2. Outlier Detection

Outliers can negatively affect distance-based algorithms such as KNN.

Outliers can be detected using the **Interquartile Range (IQR)** method:

```python
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```

Values outside the lower and upper boundaries can be investigated and, when appropriate, removed or transformed.

---

## ⚖️ Feature Scaling

KNN calculates distances between observations, so features with larger numerical ranges can dominate the distance calculation.

For this reason, feature scaling is important.

### Standardization

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted **only on the training data** to avoid data leakage.

---

## ⚠️ Handling Class Imbalance

Classification datasets can contain significantly more examples from one class than another.

Two approaches are evaluated in this project.

### 1. Random OverSampling

Random OverSampling increases the number of minority-class observations by randomly duplicating existing samples.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)

X_train_ros, y_train_ros = ros.fit_resample(
    X_train_scaled,
    y_train
)
```

### 2. SMOTE

SMOTE creates synthetic minority-class samples instead of simply duplicating existing observations.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)
```

Oversampling should be performed **only on the training data**, after splitting the dataset. The test set should remain untouched so that evaluation reflects the original data distribution.

---

## 🤖 K-Nearest Neighbors (KNN)

KNN classifies an observation based on the classes of its nearest neighbors.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_ros, y_train_ros)

y_pred = knn.predict(X_test_scaled)
```

The value of `n_neighbors` can be tuned to find an appropriate model configuration.

---

## 📊 Model Evaluation

The model can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))
```

For an imbalanced classification problem, accuracy should not be considered alone. Precision, recall, F1-score, and the confusion matrix provide additional information about performance on each class.

---

## 🔬 Experiments

The project compares KNN under different preprocessing and sampling configurations.

| Experiment                | Scaling | Sampling            |
| ------------------------- | ------- | ------------------- |
| Baseline KNN              | Yes     | None                |
| KNN + Random OverSampling | Yes     | Random OverSampling |
| KNN + SMOTE               | Yes     | SMOTE               |

The results can be compared using accuracy, precision, recall, and F1-score.

---

## 🧪 Recommended Workflow

```text
Load Dataset
     ↓
Explore Dataset
     ↓
Handle Missing Values
     ↓
Detect / Handle Outliers
     ↓
Separate Features and Target
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Apply Random OverSampling OR SMOTE
     ↓
Train KNN
     ↓
Predict Test Data
     ↓
Evaluate Model
```

### Important

The correct order is important:

1. Split the dataset into training and testing sets.
2. Fit the scaler using training data.
3. Transform training and test data.
4. Apply SMOTE or Random OverSampling **to training data only**.
5. Train KNN.
6. Evaluate on the untouched test set.

This prevents information from the test set from leaking into model training.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

---

## 📁 Project Structure

```text
Blood-Donor-Classification/
│
├── data/
│   └── blood_donor.csv
│
├── notebooks/
│   └── blood_donor_classification.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 📦 Installation

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn jupyter
```

---

## 🎯 Project Goal

The main goal is to develop a classification model that predicts blood donation behavior while investigating how preprocessing, feature scaling, outlier handling, and different oversampling techniques affect KNN classification performance.
