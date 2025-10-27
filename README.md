# 💳 Credit Card Default Risk Prediction

This project focuses on **predicting credit default risk** using machine learning classification techniques on customer demographic and financial behavior data.  
The objective is to help financial institutions identify **high-risk defaulters early** and improve **credit risk management** decisions.

---

## 🧠 Overview of Approach and Modeling Strategy

The workflow begins with **data preprocessing**, **exploratory data analysis (EDA)**, and **feature engineering**, followed by **model selection and optimization**:

- **Data Cleaning** – Addressed missing values in `AGE`, and corrected invalid negative entries in `AVG_BILL_AMT` and `PAY_TO_BILL_RATIO`.
- **Class Imbalance Handling** – Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the dataset and improve model generalization.
- **EDA & Feature Insights** – Identified key behavioral and demographic risk indicators such as:
  - Lower credit limits and higher bill amounts
  - Low `PAY_TO_BILL_RATIO` values
  - Repeated payment delays (`PAY_0 > 1`)
- **Feature Engineering** – Derived new features like:
  - `total_payment`, `credit_utilization_ratio`, and `delinquency_streak`
- **Modeling** – Compared multiple classifiers:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
  - LightGBM  

Models were evaluated using metrics such as **F2-score**, **precision**, **recall**, and **accuracy**.

---

## ⚙️ Modeling Pipeline

1. **Data Preprocessing** – Cleaning, transformation, and scaling  
2. **EDA** – Visualization with KDE and boxplots for key variables (`LIMIT_BAL`, `AGE`, `AVG_BILL_AMT`)  
3. **SMOTE Application** – To handle imbalance between defaulters and non-defaulters  
4. **Model Training & Hyperparameter Tuning** –  
   - Used **GridSearchCV** for parameter optimization  
   - **Threshold optimization** for maximizing **F2-score** and **recall**  
5. **Evaluation & Validation** – Tested on a separate validation dataset  

---

## 📊 Key EDA Insights

| Feature | Observation | Insight |
|----------|--------------|----------|
| **LIMIT_BAL** | Right-skewed | Most customers have modest credit limits |
| **AGE** | Concentrated between 25–45 | Younger & older groups more prone to default |
| **AVG_BILL_AMT** | Right-skewed with outliers | High bill amounts linked with default |
| **PAY_TO_BILL_RATIO** | Mostly low values | Underpayment is a strong risk signal |

- **Class Imbalance:** Majority are non-defaulters → mitigated with SMOTE  
- **Education vs Default:** University and graduate-level customers show higher default counts  
- **Correlation:** LIMIT_BAL and AVG_BILL_AMT moderately correlated  

---

## 🧩 Model Comparison

| Model | Strengths | Limitations | Verdict |
|-------|------------|--------------|----------|
| **Logistic Regression** | Interpretable, simple baseline | Linear assumption, low recall (51%) | ❌ Too many false negatives |
| **Decision Tree** | Easy interpretability | Overfitting, recall only 32% | ❌ Poor generalization |
| **Random Forest** | Reduced variance | Recall 38%, high computational cost | ⚠️ Moderate |
| **XGBoost** | High accuracy & precision | Low recall (32%), complex tuning | ⚠️ Precision-oriented only |
| **LightGBM** | Highest accuracy (0.84), F2-score (0.828) | Moderate recall (34%) | ✅ Best balance |

---

## 🏆 Final Model Selection

**LightGBM** was selected as the final model for its:
- Best **weighted F2-score (0.828)**  
- Balanced **precision–recall trade-off**  
- High training efficiency  
- Robust handling of large datasets  

Threshold tuning (0.60) further improved **recall** and **default detection** without major loss of precision.

---

## 💡 Business Implications

- **Delayed Payments → Early Risk Signal:** PAY_0 > 1 implies ~38% higher default risk  
- **Minimum Payment Behavior:** Low or zero payments indicate financial stress  
- **Low PAY_TO_BILL Ratios:** Strongly linked to future defaults  
- **Demographic Trends:** Younger and senior customers show higher default probabilities  
- **Marital Status:** Married customers have slightly higher default rates (~20.37%)

---

## 📈 Results Summary

| Metric | Score |
|--------|--------|
| **Accuracy** | 0.84 |
| **Weighted F2 Score** | 0.828 |
| **Precision** | 0.60 |
| **Recall (Defaulters)** | 0.34 |

---

## 🧰 Tech Stack

- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, Matplotlib, Seaborn  
- **Techniques:** SMOTE, Feature Engineering, GridSearchCV, Threshold Optimization  
- **Metrics:** F2-Score, Precision, Recall, Accuracy  

---

## 🚀 Future Improvements

- Implement **Focal Loss** to handle extreme class imbalance  
- Try **Optuna-based multi-objective optimization** for precision–recall trade-offs  
- Add **SHAP-based explainability** to enhance model interpretability  
- Deploy model via **Flask/FastAPI** for real-time risk prediction  

---

## 📬 Contact

**Author:** [Your Name]  
**GitHub:** [Your GitHub Profile Link]  
**Email:** [Your Email]  
**LinkedIn:** [Your LinkedIn Profile Link]

---

> _“Predict early. Prevent risk. Protect trust.”_
