# Telco Customer Churn Prediction

## Project Overview
This project analyzes customer churn behavior for a telecom company using the IBM Telco Customer Churn dataset.
The goal is to identify customers likely to churn and uncover the key factors driving churn, enabling the business to take proactive retention actions.

## Dataset
- **Source:** IBM Telco Customer Churn Dataset (Kaggle)
- **Records:** 7,043 customers | **Features:** 21 columns
- **Target Variable:** Churn (Yes = 26.5%, No = 73.5%)
- **Key Issue:** Class imbalance handled via `class_weight` and `scale_pos_weight`

## Tools & Technologies
- **Language:** Python 3.10
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost
- **Environment:** Jupyter Notebook | Conda (`ds_env`)

## Project Workflow

### 1. Data Cleaning
- Converted `TotalCharges` from object to float64 (contained blank spaces)
- 11 null values found in `TotalCharges` — all had `tenure = 0` (new customers), filled with `0`
- Dropped `customerID` (unique identifier, no predictive value)

### 2. Exploratory Data Analysis
- ~50% of month-to-month contract customers churn
- Fiber optic customers churn at nearly double the rate of DSL customers
- Churn is heavily concentrated in customers with tenure < 10 months
- Churned customers have higher median monthly charges (~₹80) vs retained (~₹65)

### 3. Feature Engineering
- Binary columns (Yes/No) mapped to 1/0 directly
- `gender` mapped: Male=1, Female=0
- Multi-category columns encoded using `pd.get_dummies(drop_first=True)`
- Numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) scaled using `StandardScaler`

### 4. Model Building

| Model | Recall (Churn) | Precision (Churn) | F1 (Churn) | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.56 | 0.66 | 0.60 | 0.81 |
| Random Forest | 0.49 | 0.64 | 0.55 | 0.79 |
| **XGBoost** | **0.70** | **0.54** | **0.61** | **0.76** |

- Train-test split: 80/20 with `stratify=y` to preserve class balance
- **XGBoost selected as best model** — highest recall and AUC score

### 5. Model Evaluation
- **ROC-AUC Score: 0.82** (XGBoost)
- Recall prioritized over accuracy due to business cost of missing actual churners (False Negatives)

### 6. Feature Importance (XGBoost)
| Rank | Feature | Importance |
|---|---|---|
| 1 | Contract_Two year | 0.435 |
| 2 | InternetService_Fiber optic | 0.156 |
| 3 | Contract_One year | 0.122 |
| 4 | InternetService_No | 0.060 |
| 5 | PaymentMethod_Electronic check | 0.016 |

## Key Business Insights
- **Contract type** is the strongest churn predictor — getting customers onto longer contracts is the #1 retention lever
- **Fiber optic customers** churn at alarmingly high rates — service quality or pricing may need review
- **New customers** (tenure < 10 months) are at highest risk — early engagement programs are critical
- **Electronic check payers** churn more — possibly less committed customers who haven't set up auto-pay

## Business Recommendation
> Focus retention efforts on **month-to-month fiber optic customers with high monthly charges and low tenure** — they represent the highest churn risk profile. Incentivizing customers to switch to annual or two-year contracts could significantly reduce overall churn.
