# 🔍 Fraud Detection — Assignment 1: Data Cleaning, Transformation & Feature Engineering

---

## 📌 Overview

This repository contains **Assignment 1** of my Data Science project on **Credit Card Fraud Detection**. The goal of this assignment is to prepare a synthetic transactional dataset for machine learning by performing thorough **data cleaning**, **exploratory data analysis (EDA)**, **feature engineering**, and **data transformation**.

The dataset simulates real-world financial transaction records with a strong class imbalance — only **5%** of transactions are fraudulent.

---

## 📁 Repository Structure

```
fraud-detection/
│
├── fraud_detection.ipynb        # Main Jupyter notebook (Assignment 1)
├── synthetic_fraud_dataset.csv  # Dataset used for analysis
└── README.md                    # Project documentation
```

---

## 📊 Dataset Description

| Property | Details |
|---|---|
| Total Records | 10,000 transactions |
| Total Features | 10 columns |
| Fraud Cases | 500 (5%) |
| Non-Fraud Cases | 9,500 (95%) |
| Amount Range | ₹1 — ₹11,628 |
| Average Amount | ₹178.14 |

### Feature Overview

| Column | Type | Description |
|---|---|---|
| `transaction_id` | int | Unique transaction identifier |
| `user_id` | int | Unique user identifier |
| `amount` | float | Transaction amount |
| `transaction_type` | str | ATM / QR / Online / POS |
| `merchant_category` | str | Travel / Food / Clothing / Grocery / Electronics |
| `country` | str | TR / US / FR / DE / UK / NG |
| `hour` | int | Hour of the day (0–23) |
| `device_risk_score` | float | Risk score from device fingerprint |
| `ip_risk_score` | float | Risk score from IP address |
| `is_fraud` | int | Target label — 0 (legitimate) / 1 (fraud) |

---

## 🧹 Step 1 — Data Cleaning & Inspection

- Verified dataset shape: **(10,000 rows × 10 columns)**
- Checked for **null values** — none found
- Checked for **duplicate records** — none found
- Inspected data types using `df.info()` and `df.describe()`
- Reviewed random samples to understand data quality

---

## 📈 Step 2 — Exploratory Data Analysis (EDA)

Performed visual analysis using **Seaborn** and **Matplotlib** to uncover patterns in fraud behaviour:

| Visualization | Insight |
|---|---|
| `countplot` — is_fraud | Confirmed heavy class imbalance (95:5) |
| `countplot` — transaction_type vs fraud | Online transactions show higher fraud rates |
| `countplot` — merchant_category vs fraud | Travel & Electronics categories more prone to fraud |
| `countplot` — country vs fraud | Fraud concentrated in specific geographies |
| `histplot` — amount vs fraud | Fraudulent transactions often involve higher amounts |
| `boxplot` — hour vs fraud | Night-time hours show elevated fraud activity |
| `boxplot` — amount vs fraud | Fraud transactions have a wider and higher amount range |
| `heatmap` — correlation matrix | device_risk_score and ip_risk_score are strong predictors |
| `lineplot` — hour vs amount | Fraud spikes at specific hours of the day |
| `scatterplot` — device_risk_score vs amount | High-risk devices cluster with high-value fraud transactions |

---

## 🛠️ Step 3 — Feature Engineering

Created a new feature `time_period` by binning the `hour` column into four meaningful time segments:

```python
def classify_time(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df['time_period'] = df['hour'].apply(classify_time)
```

**Key Finding:** Night-time transactions show a disproportionately higher rate of fraud, making `time_period` a meaningful engineered feature.

---

## 🔄 Step 4 — Data Transformation

Applied **one-hot encoding** to all categorical columns to convert them into numeric format suitable for machine learning:

```python
df_encoded = pd.get_dummies(
    df,
    columns=["transaction_type", "merchant_category", "country", "time_period"],
    drop_first=True   # avoids multicollinearity
)
```

- `drop_first=True` was used to prevent the **dummy variable trap**
- Post-encoding correlation heatmap was generated to inspect feature relationships in the transformed dataset

---

## 🧰 Tech Stack

- **Python 3.10+**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Seaborn** — statistical visualizations
- **Matplotlib** — plotting
- **Jupyter Notebook** — interactive development

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install pandas numpy seaborn matplotlib jupyter
```

### 3. Launch the notebook
```bash
jupyter notebook fraud_detection.ipynb
```

---

## 💡 Key Takeaways

- **Class imbalance** is a critical challenge — only 5% of transactions are fraud. This will need to be addressed in the modelling phase (SMOTE, class weights, etc.)
- **Device and IP risk scores** are strong numeric indicators of fraud
- **Time of day** matters — the engineered `time_period` feature captures temporal fraud patterns
- **One-hot encoding** significantly expands the feature space but also adds useful signal from categorical variables

---

## 🔮 What's Next — Assignment 2

- Build baseline ML models (Logistic Regression, Decision Tree, Random Forest)
- Handle class imbalance using SMOTE or class weighting
- Evaluate models using Precision, Recall, F1-Score, and ROC-AUC
- Tune hyperparameters for optimal performance

---

## 👤 Author

**Your Name**
- LinkedIn: [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
- GitHub: [github.com/your-username](https://github.com/your-username)

---

## 📄 License

This project is for educational purposes. The dataset is synthetically generated and does not contain any real financial data.
