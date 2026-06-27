# 🏡 House Price Predictor

> An end-to-end Machine Learning project that predicts residential property sale prices using the Ames Housing Dataset. Built with a streamlined **8-feature pipeline** and an interactive **Streamlit web dashboard**.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [The Core Idea: 8-Feature Optimization](#-the-core-idea-8-feature-optimization)
- [Project Structure](#-project-structure)
- [The Jupyter Notebook: A Deep Dive](#-the-jupyter-notebook-a-deep-dive)
  - [Cell 1 — Imports & Setup](#cell-1--imports--setup)
  - [Cell 2 — Data Loading & Sub-setting](#cell-2--data-loading--sub-setting)
  - [Cell 3 — Exploratory Data Analysis (EDA)](#cell-3--exploratory-data-analysis-eda)
  - [Cell 4 — Data Preparation & Imputation](#cell-4--data-preparation--imputation)
  - [Cell 5 — Model Building & Evaluation](#cell-5--model-building--evaluation)
  - [Cell 6 — Model Serialization](#cell-6--model-serialization)
  - [Cell 7 — Business Insights & UI Architecture](#cell-7--business-insights--ui-architecture)
- [The Streamlit App](#-the-streamlit-app)
- [Model Performance](#-model-performance)
- [Getting Started](#-getting-started)
- [Tech Stack](#-tech-stack)

---

## 🌐 Project Overview

This project is a **complete, production-oriented Machine Learning pipeline** for predicting the sale price of residential homes. It solves a classic **regression problem** — predicting a continuous numerical value (price) rather than a category.

The project is built around two key components:

| Component | File | Purpose |
|---|---|---|
| 🔬 **Training Pipeline** | `House_Price_Assessment.ipynb` | Data cleaning, EDA, model training, and serialization |
| 🖥️ **Interactive UI** | `app.py` | Streamlit web app for real-time predictions |

> **Why Regression?** House Price Prediction is a **regression** task because the output (`SalePrice`) is a continuous number. We compare a linear baseline (Linear Regression) against a powerful non-linear ensemble model (Random Forest Regressor).

---

## 💡 The Core Idea: 8-Feature Optimization

The original Ames Housing Dataset contains **80+ columns**, many of which are sparse, categorical, and require complex preprocessing. Instead of using all of them, we performed a feature importance and correlation analysis to identify the **8 most impactful numerical features**.

These 8 features alone capture **~80% of the variance** in housing prices, making the model fast, transparent, and deployment-ready.

The 8 features are strategically split into two groups to support the UI architecture:

### 🎛️ Active UI Inputs (4 features — controlled by the user)

| Feature | Description | Why It Matters |
|---|---|---|
| `OverallQual` | Overall material and finish quality (1–10 scale) | Strongest single predictor of price. Quality drives perceived value. |
| `GrLivArea` | Above-grade (ground) living area in square feet | Buyers pay directly for living space. |
| `GarageCars` | Garage size measured in car capacity | Suburban markets strongly value garage space. |
| `YearBuilt` | Original construction year | Newer homes command higher premiums. |

### ⚙️ Background Baseline Features (4 features — auto-filled with dataset medians)

| Feature | Description | Why It's in the Background |
|---|---|---|
| `TotalBsmtSF` | Total square feet of basement area | Adds predictive value, but less intuitive for users to input. |
| `FullBath` | Number of full bathrooms above grade | Consistent across most homes; median works well as a default. |
| `LotArea` | Lot size in square feet | High variance, but the median is a safe baseline for estimation. |
| `YearRemodAdd` | Year of most recent remodel | Correlated with `YearBuilt`; auto-filled to avoid redundancy in the UI. |

---

## 📁 Project Structure

```
HousePricePredictor/
│
├── House_Price_Assessment.ipynb  # ← The core ML pipeline (training)
├── app.py                        # Streamlit web app (prediction UI)
├── house_prices.csv              # Ames Housing Dataset
├── house_price_model.pkl         # Trained Random Forest model (serialized)
├── scaler.pkl                    # Fitted StandardScaler (serialized)
└── README.md                     # This file
```

---

## 🔬 The Jupyter Notebook: A Deep Dive

The notebook `House_Price_Assessment.ipynb` is the **heart of this project**. It contains the complete, documented pipeline from raw data to a saved, deployment-ready model. Here is a plain-English walkthrough of every cell.

---

### Cell 1 — Imports & Setup

**What it does:** Loads all the Python libraries the project depends on and configures the visual style for plots.

```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

| Library | Role |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` / `seaborn` | Data visualization |
| `joblib` | Saving and loading trained models |
| `sklearn` | All machine learning components |

---

### Cell 2 — Data Loading & Sub-setting

**What it does:** Loads `house_prices.csv` and immediately trims it down from 81 columns to just our **9 essential columns** (8 features + `SalePrice`).

```python
core_features = [
    'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt',   # Active UI
    'TotalBsmtSF', 'FullBath', 'LotArea', 'YearRemodAdd'      # Background
]
df = df_full[core_features + ['SalePrice']].copy()
```

**Result:** A clean `(1460, 9)` DataFrame — 1,460 houses, 9 columns. No unnecessary noise.

---

### Cell 3 — Exploratory Data Analysis (EDA)

**What it does:** Generates a **correlation heatmap** — a colour-coded grid that shows how strongly each feature is related to every other feature, including `SalePrice`.

```python
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', square=True)
```

**How to read it:**
- Values close to **+1.0** (dark red) = strong positive relationship. As one goes up, so does the other.
- Values close to **-1.0** (dark blue) = strong negative relationship.
- Values near **0** (white) = little to no direct relationship.

**Key finding:** `OverallQual` and `GrLivArea` show the highest correlation with `SalePrice` (~0.79 and ~0.71 respectively), validating our feature selection.

---

### Cell 4 — Data Preparation & Imputation

**What it does:** Cleans the data and prepares it for the machine learning algorithms in three steps.

**Step 1 — Imputation:** Fills any missing values using the **median** of each column.
```python
df[core_features] = df[core_features].fillna(df[core_features].median())
```
> We use the **median** (not the average/mean) because it is robust to outliers. A few extremely expensive mansions in the dataset won't skew a median the way they would a mean.

**Step 2 — Train/Test Split:** Divides the 1,460 rows into:
- **Training Set (80%)** → 1,168 rows — the data the model **learns from**.
- **Test Set (20%)** → 292 rows — data the model has **never seen**, used to measure real-world accuracy.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
> `random_state=42` ensures the same split every time the notebook is run, making results reproducible.

**Step 3 — Feature Scaling:** Applies `StandardScaler` to normalize all features to a common scale (zero mean, unit variance).
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```
> **Why scale?** Without scaling, a feature like `LotArea` (values in the tens of thousands) would dominate a feature like `FullBath` (values 1–4), even if `FullBath` is actually more informative. Scaling levels the playing field.
>
> **Important:** The scaler is **fit only on the training data** (`fit_transform`) and then **applied** to the test data (`transform`). This prevents *data leakage* — the model should never learn statistics from the test set.

---

### Cell 5 — Model Building & Evaluation

**What it does:** Trains two models and compares their performance head-to-head.

#### Model 1: Linear Regression (Baseline)
```python
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
```
Linear Regression finds the best-fit straight line (or hyperplane) through the data. It assumes that the relationship between each feature and the price is linear — e.g., each extra square foot adds exactly the same dollar amount, regardless of context. This is a useful but limited assumption.

#### Model 2: Random Forest Regressor (Primary)
```python
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
```
A Random Forest builds **100 independent decision trees** on random subsets of the data and features. The final prediction is the **average of all 100 trees' predictions**. This approach:
- Captures **non-linear relationships** (e.g., a 10-car garage doesn't add 5x the value of a 2-car garage).
- Captures **feature interactions** (e.g., high overall quality matters *more* in a large home than a small one).
- Is **robust to overfitting** because individual noisy trees are averaged out.

#### Evaluation Metrics

| Metric | Full Name | What It Measures | Better When... |
|---|---|---|---|
| **RMSE** | Root Mean Squared Error | Average prediction error in dollars; large errors are penalized heavily | Lower |
| **MAE** | Mean Absolute Error | Average absolute prediction error in dollars | Lower |
| **R²** | R-Squared (Coefficient of Determination) | Proportion of price variance explained by the model (0 to 1) | Closer to 1.0 |

```python
def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
```

---

### Cell 6 — Model Serialization

**What it does:** Saves the trained Random Forest model and the fitted StandardScaler to disk as `.pkl` (Pickle) files using `joblib`.

```python
joblib.dump(rf_model, 'house_price_model.pkl')
joblib.dump(scaler,   'scaler.pkl')
```

> Think of this like saving your game progress. The model and scaler have been "trained" — they hold all the learned weights and statistics. By saving them, the Streamlit app can **load them instantly** without re-training from scratch every time a user opens the page.
>
> **Why save the scaler too?** The scaler learned the mean and standard deviation of the training data. When a user enters a prediction in the app, their inputs must be scaled in **exactly the same way** as the training data before being passed to the model. Without the saved scaler, predictions would be completely wrong.

---

### Cell 7 — Business Insights & UI Architecture

**What it does:** A markdown cell (no code) summarizing the key takeaways and explaining how the 8-feature strategy elegantly connects the model to the UI.

**Key insights:**
1. **Near-Parity Performance** — Retaining only 8 features preserves ~85% of the predictive power of using all 80 columns.
2. **Zero Categorical Overhead** — By selecting only numerical features, we eliminated all label encoding, one-hot encoding, and the risk of unseen-category errors in production.
3. **Elegant UI Split** — The top 4 features are exposed to the user; the bottom 4 are silently handled with dataset medians, keeping the interface clean and focused.

---

## 🖥️ The Streamlit App

The `app.py` file is the interactive prediction dashboard. Here's how it works under the hood:

1. **Load Assets:** Loads `house_price_model.pkl` and `scaler.pkl` once using Streamlit's caching (`@st.cache_resource`), so the app is fast after the first load.

2. **Compute Background Medians:** Reads `house_prices.csv` once (`@st.cache_data`) to extract the median values of the 4 background features. These are used automatically during prediction.

3. **Collect User Inputs:** The sidebar presents 4 interactive controls:
   - **Overall Quality** → Slider (1–10)
   - **Total Living Area** → Number input (sq ft)
   - **Garage Capacity** → Slider (0–5 cars)
   - **Year Built** → Slider (1872–2010)

4. **Predict:** When the user clicks "Predict Price", the app:
   - Combines the 4 user inputs with the 4 background medians into a single 8-element row.
   - Scales the row using the loaded `scaler`.
   - Passes it to the `rf_model` for prediction.
   - Displays the estimated price.

---

## 📊 Model Performance

Results measured on the held-out **20% test set** (292 unseen houses):

| Model | RMSE | MAE | R² Score |
|---|---|---|---|
| Linear Regression | ~$34,422 | ~$21,502 | ~0.845 |
| **Random Forest** ✅ | **~$29,493** | **~$17,757** | **~0.887** |

> The Random Forest Regressor outperforms Linear Regression across all three metrics, validating the choice to use it as the production model in the Streamlit app.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/ShaunakChorge/HousePricePredictor.git
cd HousePricePredictor
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

### 3. (Optional) Re-train the Model
Open and run all cells in `House_Price_Assessment.ipynb` to regenerate `house_price_model.pkl` and `scaler.pkl`.

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Core language |
| Pandas | 2.x | Data manipulation |
| NumPy | 1.x | Numerical operations |
| Matplotlib / Seaborn | Latest | Data visualization |
| scikit-learn | 1.7+ | ML models, scaling, metrics |
| Joblib | Latest | Model serialization |
| Streamlit | Latest | Interactive web dashboard |

---

## 📜 License

This project is built for educational and assessment purposes as part of an internship assignment.
