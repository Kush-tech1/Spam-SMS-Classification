
# 📧 SMS Spam Detection Using Machine Learning

A complete end-to-end machine learning pipeline to classify SMS messages as **Spam** or **Ham** using NLP techniques and multiple ML algorithms.

---

## 🚀 Project Overview

This project builds a robust SMS spam classifier using Python, NLP preprocessing, TF-IDF vectorization, and various machine learning models.
It includes:

✔ Data Cleaning
✔ Text Preprocessing (tokenization, stemming, stopword removal)
✔ EDA (distribution, word clouds, histograms)
✔ Feature Engineering
✔ Model Training & Evaluation
✔ Ensemble Learning (Voting & Stacking Classifiers)
✔ Saving vectorizer & model (`vectorizer.pkl`, `model.pkl`)

---

## 📂 Dataset

The dataset used: **spam.csv**
Contains two main columns:

* **v1** → target label (ham/spam)
* **v2** → message text

After cleaning:

* **5169 rows**
* **2 meaningful columns**

  * `target` (0 = ham, 1 = spam)
  * `text` (SMS message)

---

## 🧹 Data Cleaning

* Removed unused columns (`Unnamed: 2, 3, 4`)
* Renamed columns → `target`, `text`
* Encoded labels (ham=0, spam=1) using `LabelEncoder`
* Removed **403 duplicate entries**
* Final cleaned dataset → **5169 messages**

---

## 📊 Exploratory Data Analysis

Performed:

* Target distribution (pie chart)
* Histogram of text characteristics

  * number of characters
  * number of words
  * number of sentences
* Pairplots & heatmaps
* Word clouds

  * Spam messages word cloud
  * Ham messages word cloud
* Most common words (spam/ham)

Key insight:
⚠ **Dataset is imbalanced** — spam = 653 messages, ham = 4516.

---

## 🧠 NLP Preprocessing

Custom `transform_text()` pipeline:

1. Lowercase
2. Tokenization
3. Keep alphanumeric tokens
4. Remove stopwords and punctuation
5. Stemming using **PorterStemmer**

Created new column:
✔ `transformed_text`

---

## 🧩 Feature Engineering

Generated additional features:

* `num_characters`
* `num_words`
* `num_sentences`

### Text Vectorization

Used:

* **TF-IDF Vectorizer** (`max_features=3000`)
* Output shape: `(5169, 3000)`

---

## 🤖 Model Building

Trained multiple ML models:

### **Naive Bayes**

* GaussianNB
* MultinomialNB
* BernoulliNB

### **Other ML Algorithms**

* Logistic Regression
* SVC
* Decision Tree
* KNN
* Random Forest
* ExtraTrees
* AdaBoost
* Bagging Classifier
* Gradient Boosting
* XGBoost

---

## 🏆 Model Performance

Best performing models based on **precision for spam detection**:

| Model                      | Accuracy | Precision |
| -------------------------- | -------- | --------- |
| **MultinomialNB**          | 0.9719   | **1.00**  |
| **KNN**                    | 0.900    | 1.00      |
| **ExtraTreesClassifier**   | 0.9777   | 0.9915    |
| **RandomForestClassifier** | 0.9700   | 0.9908    |

---

## 🤝 Ensemble Learning

### ✔ Voting Classifier (soft voting)

**Final Model Performance**

* **Accuracy: 0.9816**
* **Precision: 0.9917**

This was saved as the final deployment model.

### ✔ Stacking Classifier

Also tested but voting performed better.




Just tell me!
