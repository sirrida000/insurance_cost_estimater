# 🩺 Insurance Cost Estimator

A machine learning project that predicts **annual medical insurance charges** from a person's details (age, sex, BMI, children, smoker status and region), with a simple **Streamlit web app** to try it out.

---

## 📌 Project Overview

Medical insurance costs depend on many personal factors. In this project I:

1. Cleaned the data (removed duplicate rows)
2. Explored it with plots, skewness, kurtosis and outlier checks
3. Fixed the skewed target (`charges`) with a log transform
4. Trained and compared three regression models
5. Built a Streamlit app that gives a live cost estimate

This is a **regression** problem, because the model predicts a number (the charge).

---

## 📊 Dataset

- **File:** `medical_insurance.csv`
- **Rows:** 2,772 in the raw file → **1,337** after removing duplicates
- **Target:** `charges` (annual insurance cost)

| Column | Type | Description |
|---|---|---|
| age | number | Age of the person (18 to 64) |
| sex | category | male / female |
| bmi | number | Body mass index |
| children | number | Number of children (0 to 5) |
| smoker | category | yes / no |
| region | category | northeast / northwest / southeast / southwest |
| charges | number | Annual medical insurance cost (target) |

---

## 🧹 Data Cleaning

- Every row in the raw file appeared **at least twice** (some four times), so the dataset had been duplicated.
- Removed **1,435 duplicate rows** with `df.drop_duplicates()`, leaving **1,337 unique rows**.

---

## 🔍 Exploratory Data Analysis

| Column | Skew | Kurtosis | What it shows |
|---|---|---|---|
| age | 0.055 | -1.244 | Balanced and flat, ages spread evenly |
| bmi | 0.284 | -0.053 | Almost a perfect bell curve |
| children | 0.937 | 0.201 | Slightly right-heavy |
| charges | 1.515 | 1.604 | Strongly right-skewed with heavy tails |

**Key findings**

- `charges` has a long right tail. The mean (about 13,279) is much higher than the median (about 9,386).
- **Smoking is the biggest driver of cost.** The average charge for smokers is about **32,050** versus about **8,441** for non-smokers.
- About **94%** of people with charges above 30,000 are smokers. They form the second bump in the charges histogram.
- Outlier check (1.5 × IQR rule): `bmi` has only 9 outliers (0.7%), while `charges` has 139 (10.4%). These are real people, so the rows were **kept**.

---

## ⚙️ Feature Engineering

- **Log transform of the target:** `log_charges = np.log1p(charges)` reduced the skew from **1.515 to about -0.09**.
- **One-hot encoding** of `sex`, `smoker` and `region`.
- **BMI category** (underweight / normal / overweight / obese) added as an extra feature.
- Model predictions are converted back to real money with `np.expm1()`.

---

## 🤖 Models and Results

Three regression models were trained and compared on a held-out test set:

| Model | R² | MAE | RMSE |
|---|---|---|---|
| **Random Forest** | **0.8244** | **2407.39** | **5169.91** |
| Gradient Boosting | 0.8159 | 2432.43 | 5292.51 |
| Linear Regression | 0.5408 | 4489.72 | 8359.39 |

**Best model: Random Forest.** It explains about 82% of the variation in charges.

Linear Regression did worst because the relationship is not a single straight line: smokers and non-smokers follow very different patterns. Tree-based models handle this much better.

---

## 🖥️ Streamlit App

The app lets a user enter patient details and get an instant estimate:

- Sliders and buttons for age, sex, BMI, children, smoker and region
- An **Estimate charges** button
- The predicted annual charge shown as a large number
- A bar chart comparing the prediction with the **average non-smoker** and **average smoker** charges

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy** for data handling
- **Matplotlib, Seaborn** for visualisation
- **Scikit-learn** for modelling
- **Joblib** for saving the model
- **Streamlit** for the web app
- **Jupyter Notebook** for analysis

---

## 📁 Project Structure
```
insurance-cost-prediction/
│
├── app.py                  # Streamlit app
├── project.ipynb           # Data cleaning, EDA and model training
├── medical_insurance.csv   # Dataset
├── insurance_model.pkl     # Saved best model (Random Forest)
├── model_features.pkl      # Feature column names the model was trained on
├── model_name.pkl          # Name of the best model
└── README.md
```

> Update the file names above to match your own folder.

---

## 🚀 How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/sirrida000/insurance-cost-prediction.git
   cd insurance-cost-prediction
   ```

2. **Install the libraries**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
   ```

3. **Run the app**

   ```bash
   streamlit run app.py
   ```

---

## 💡 What I Learned

- Finding and removing duplicate data
- Reading skewness, kurtosis and outliers, and what they mean for a model
- Why a log transform helps with skewed targets
- Training, comparing and choosing between regression models
- Avoiding data leakage by keeping the target out of the features
- Turning a trained model into a working web app

---

## 🔮 Future Improvements

- Hyperparameter tuning (GridSearchCV)
- Cross-validation for more reliable scores
- Try XGBoost / LightGBM
- Show feature importance in the app
- Deploy on Streamlit Community Cloud

---

## 👩‍💻 Author

**Reeda Shaikh**
BTech student in AI and Data Science

- GitHub: [sirrida000](https://github.com/sirrida000)
- LinkedIn: [linkedin.com/in/reeda-s-11914a42b](https://linkedin.com/in/reeda-s-11914a42b)
