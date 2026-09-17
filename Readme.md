# Social Sentiment Analysis for F&B

> **Data Analytics Midterm Project**
> Social Media Sentiment Analysis Pipeline with Machine Learning & Streamlit Dashboard

## 📌 Overview

This project develops a social media sentiment analysis pipeline for a simulated F&B product launch scenario.

The project applies four layers of analytics:

- **Descriptive Analytics** – Explore data characteristics and sentiment distribution.
- **Diagnostic Analytics** – Investigate keywords, topics, and relationships within the data.
- **Predictive Analytics** – Build and evaluate machine learning models for sentiment classification.
- **Prescriptive Analytics** – Translate analytical results into actionable recommendations.

The project also includes an interactive **Streamlit dashboard** for exploring analytical results.

> **Note:** The dataset used is Sentiment140, a general English Twitter dataset. It does not represent actual customer mentions of an F&B product. Therefore, the F&B context is used as a simulated application scenario.

---

## 🎯 Objectives

The main objectives of this project are to:

1. Understand and preprocess social media text data.
2. Explore sentiment patterns and data characteristics.
3. Identify important keywords and latent topics.
4. Build sentiment classification models using TF-IDF features.
5. Compare different machine learning algorithms.
6. Evaluate model performance using statistical and classification metrics.
7. Develop an interactive dashboard to support data exploration.
8. Translate analytical findings into practical recommendations.

---

## 📊 Dataset

The project uses the **Sentiment140** dataset containing English Twitter posts labeled with two sentiment classes:

| Label | Sentiment |
| ----- | --------- |
| 0     | Negative  |
| 4     | Positive  |

For computational efficiency, **60,000 tweets** were used in the analysis.

The dataset is originally a general-purpose Twitter sentiment dataset collected in 2009 and is not specific to the F&B industry.

---

## 🔄 Analytics Pipeline

```text
Raw Twitter Data
       │
       ▼
Data Cleaning & Preprocessing
       │
       ▼
Descriptive Analytics
       │
       ▼
Diagnostic Analytics
       │
       ▼
Predictive Analytics
       │
       ▼
Prescriptive Analytics
       │
       ▼
Streamlit Dashboard
```

---

## 🤖 Predictive Modeling

Three machine learning algorithms were evaluated:

- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Linear SVM**

Text features were extracted using **TF-IDF** with:

- `max_features = 10,000`
- `ngram_range = (1, 2)`
- `min_df = 3`

The dataset was split into:

- **80% training set:** 48,000 tweets
- **20% test set:** 12,000 tweets

A stratified split was used to maintain the class distribution.

### Model Performance

| Model                   |   Accuracy |  Precision |     Recall |   F1-score |
| ----------------------- | ---------: | ---------: | ---------: | ---------: |
| Multinomial Naive Bayes |     76.27% |     77.46% |     74.10% |     75.74% |
| Logistic Regression     | **78.42%** | **78.28%** | **78.67%** | **78.47%** |
| Linear SVM              |     77.08% |     76.65% |     77.90% |     77.27% |

Logistic Regression achieved the highest Accuracy and F1-score among the three evaluated models, with an **AUC of 0.863**.

---

## 🔎 Diagnostic Analysis

The diagnostic layer includes:

- Sentiment distribution analysis
- Keyword analysis
- Topic modeling using **Latent Dirichlet Allocation (LDA)**
- Examination of relationships between time-related variables and sentiment labels
- Error analysis of model predictions

These analyses help identify patterns that cannot be observed from overall sentiment percentages alone.

---

## 📈 Dashboard

An interactive dashboard was developed using **Streamlit**.

The dashboard allows users to explore analytical results using filters such as:

- Date range
- Sentiment
- Topic

It provides a more accessible way to inspect the results without directly working with the analysis notebook.

---

## 💡 Prescriptive Insights

Based on the analytical results, the proposed workflow includes:

1. Use Logistic Regression as an initial sentiment filtering model.
2. Prioritize strongly negative mentions for further review.
3. Combine automated classification with manual review for short, sarcastic, ambiguous, or context-dependent posts.
4. Monitor sentiment trends through the dashboard.
5. Analyze sentiment by topic instead of relying only on an overall negative percentage.

These recommendations are intended as a **simulated deployment scenario** and should be re-evaluated using real F&B customer mentions before operational deployment.

---

## 📁 Project Structure

```text
social-sentiment-fnb/
│
├── data/
│   └── ...
│
├── notebook/
│   └── Midterm_DA.ipynb
│
├── dashboard/
│   ├── app.py
│   └── ...
│
├── report/
│   └── Midterm_DA.docx
│
├── slides/
│   └── ...
│
├── README.md
└── requirements.txt
```

> Folder and file names may vary depending on the final project structure.

---

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/social-sentiment-fnb.git
cd social-sentiment-fnb
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open the analysis notebook:

```text
notebook/Midterm_DA.ipynb
```

### 4. Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will then be available locally in your browser.

---

## ⚠️ Limitations

Several limitations should be considered:

- The dataset is a general Twitter dataset rather than real F&B customer feedback.
- Only two sentiment classes are available: Positive and Negative.
- The TF-IDF approach has limitations when handling sarcasm, short texts, and missing context.
- Temporal relationships in the dataset require caution when interpreting sentiment trends.
- Results should be validated again using real-world F&B social media data before deployment.

---

## 👥 Team

| Member                    | Main Contribution                                             |
| ------------------------- | ------------------------------------------------------------- |
| **Trương Huỳnh Hoài Bảo** | Team coordination, data processing, and overall result review |
| **Nguyễn Hoàng Minh**     | Data analysis and Streamlit dashboard development             |
| **Chung Nguyễn Minh Trí** | Data preprocessing and four-layer analytics pipeline          |
| **Đặng Vĩnh Quang**       | Result interpretation and presentation preparation            |
| **Lê Trung Bảo**          | Result synthesis and report preparation                       |

All members contributed to reviewing and validating the overall project.

---

## 📚 References

1. A. Go, R. Bhayani, and L. Huang, _Twitter Sentiment Classification using Distant Supervision_, Stanford University, 2009.
2. Sentiment140 Dataset.
3. F. Pedregosa et al., _Scikit-learn: Machine Learning in Python_, JMLR, 2011.
4. D. M. Blei, A. Y. Ng, and M. I. Jordan, _Latent Dirichlet Allocation_, JMLR, 2003.
5. Streamlit Documentation.

---

## 📄 Project Context

This repository was developed as part of a **Data Analytics Midterm Project**.

The project demonstrates an end-to-end workflow from **data preparation and exploratory analysis to machine learning, visualization, and actionable recommendations**.
