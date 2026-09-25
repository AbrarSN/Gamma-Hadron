# MAGIC Gamma Telescope Classification

This repository contains data preprocessing, exploratory analysis, feature scaling, and machine learning classification models (Gaussian Naive Bayes and K-Nearest Neighbors) applied to the MAGIC Gamma Telescope dataset[cite: 3].

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTu-Hjt3SK03uGnqeChpiaExzYkte1Wer2Ase-7zyaD2w&s=10">

## introduction :
Gamma-Hadron Prediction. It is related to high-energy physics and astronomy.
When high-energy particles enter the Earth's atmosphere, they create particle showers. Our project analyzes the data produced by these showers and tries to identify whether an event is caused by a gamma ray or a hadron.
For this, we use machine learning. First, we collect and preprocess the data. Then, we train a machine-learning model using known gamma and hadron events. After training, the model can predict the category of new events.

---
## 🎯 Project Objective

The primary objective of this project is to develop and evaluate a machine learning classification model capable of distinguishing between high-energy gamma particles (signal) and hadron particles (background). By analyzing the simulated geometric parameters of atmospheric Cherenkov radiation showers, the model aims to accurately classify the origin of the primary particle. This classification assists in high-energy astrophysical observations by effectively filtering out background hadronic noise from actual gamma-ray signals.

**Key Goals:**
*   Perform exploratory data analysis (EDA) on the Cherenkov photon shower parameters (e.g., length, width, size, concavity).
*   Train and optimize binary classification algorithms (such as Random Forest, SVM, or Neural Networks) to accurately separate gamma rays from hadronic showers.
*   Evaluate model performance using critical metrics like Accuracy, Precision, Recall, and ROC-AUC to minimize false positives.

## 🔗 References

*   **Kaggle Dataset:** [MAGIC Gamma Telescope Dataset](https://www.kaggle.com/datasets/fedesoriano/magic-gamma-telescope-dataset) *(Replace with your specific Kaggle link if different)*
*   **GitHub Repository:** [Project Source Code](https://github.com/your-username/your-repo-name) *(Replace with your actual GitHub repository link)*

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR13MOEybq6L1kJhaX90AWTE4D7LsemOWHxw5qf-Aucww&s">

## Why is it useful?
Separating gamma rays from hadrons is important because gamma rays can provide information about high-energy objects and processes in the universe. Machine learning can help automate this classification and reduce the amount of manual analysis required.


## 📌 Dataset Overview

* **Total Samples**: 19,020 entries[cite: 3]
* **Target Variable**: `class`[cite: 3]
  * `g` (Gamma rays) mapped to `1`[cite: 3]
  * `h` (Hadron noise) mapped to `0`[cite: 3]
* **Numerical Features (10)**: `fLength`, `fWidth`, `fSize`, `fConc`, `fConc1`, `fAsym`, `fM3Long`, `fM3Trans`, `fAlpha`, `fDist`[cite: 3]

---

## ⚙️ Data Preprocessing Pipeline

1. **Train/Val/Test Split**: Split into Training (60%), Validation (20%), and Testing (20%) sets[cite: 3].
2. **Feature Scaling**: Scaled feature distributions using `StandardScaler`[cite: 3].
3. **Class Balancing**: Balanced training class distributions using `RandomOverSampler`[cite: 3].

---

## 📊 Model Training & Performance

### 1. Gaussian Naive Bayes (`GaussianNB`)
* **Test Accuracy**: 71% (evaluated on 3,804 test samples)[cite: 3]
* **Classification Breakdown**:
  * **Class 0.0 (Hadron)**: Precision: 0.67 | Recall: 0.38 | F1-Score: 0.49[cite: 3]
  * **Class 1.0 (Gamma)**: Precision: 0.72 | Recall: 0.89 | F1-Score: 0.80[cite: 3]

### 2. K-Nearest Neighbors (`KNeighborsClassifier`)
* Initialized with `n_neighbors=3` for classification comparisons[cite: 3].

The Gamma-Hadron Prediction project demonstrates the application of machine learning techniques in the classification of high-energy particle events. By analyzing particle-shower characteristics and identifying patterns within the data, the model can distinguish between gamma-ray and hadron events. This approach can contribute to more efficient and accurate analysis of cosmic-ray data. Overall, the project highlights the potential of machine learning in high-energy physics and astrophysics, particularly in supporting the identification and study of gamma-ray events.

---

## 🚀 Setup & Execution

### Dependencies
Ensure you have the following Python packages installed:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
