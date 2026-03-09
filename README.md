# Customer Churn Prediction with Ensemble Learning

A machine learning project comparing and combining gradient boosting methods (XGBoost, LightGBM, CatBoost) with stacking ensemble for customer churn prediction.

## Project Overview

This project implements a complete ML pipeline for predicting customer churn using a synthetic dataset of 100,000 customers. It demonstrates:

- **Classical ML**: Optimized gradient boosting with stacking ensemble
- **Ensemble Learning**: Stacking approach combining multiple models

- **Hyperparameter Optimization using Optuna**

## Dataset Description

The project uses a synthetic customer churn dataset of 100,000 customers.  
The dataset can be accessed and downloaded from Kaggle:

[Telco Customer Churn Dataset](https://www.kaggle.com/datasets/dhrubangtalukdar/telco-customer-churn-data)


| Feature | Type | Description |
|---------|------|-------------|
| CustomerID | Integer | Unique identifier (removed in preprocessing) |
| Age | Integer | Customer age in years |
| Gender | Categorical | Male/Female |
| Tenure | Integer | Months with company |
| MonthlyCharges | Float | Monthly fee ($) |
| Contract | Categorical | Month-to-month, One year, Two year |
| PaymentMethod | Categorical | Electronic check, Mailed check, Bank transfer, Credit card |
| TotalCharges | Float | Cumulative charges ($) |
| **Churn** | **Binary Target** | **Yes/No** - Customer left in last month |

**Dataset Size**: 100,000 customers  
**Churn Rate**: ~33%

## Machine Learning Pipeline

The project follows a structured machine learning workflow composed of four main stages.

### 1️⃣ Data Loading & Exploration
**Notebook:** `01_load_data.ipynb`

This stage focuses on understanding the dataset before building models. The main steps include:

- Dataset inspection
- Distribution analysis of features
- Churn class imbalance analysis
- Initial data visualization

---

### 2️⃣ Data Preprocessing
**Notebook:** `02_preprocessing.ipynb`

The preprocessing stage prepares the dataset for machine learning models.

Key steps include:

- Removing non-informative features (`CustomerID`)
- Encoding categorical variables
- Splitting the dataset into training and testing sets
- Applying feature scaling 
- Saving processed datasets for reuse

---

### 3️⃣ Hyperparameter Optimization
**Notebook:** `03_optuna_hyperparameter_tuning.ipynb`

Model hyperparameters are optimized using **Optuna** to improve predictive performance.

The following models are optimized:

- XGBoost  
- LightGBM  
- CatBoost  

The optimization process searches for the best combination of:

- Learning rate
- Number of estimators
- Tree depth
- Regularization parameters

---

### 4️⃣ Model Training & Ensemble Learning
**Notebook:** `04_models_training.ipynb`

This notebook trains the optimized models and constructs a **stacking ensemble**.

**Base models:**

- XGBoost  
- LightGBM  
- CatBoost  

**Meta-learner:**

- Logistic Regression

Stacking improves predictive performance by combining the strengths of multiple models.

---

## Model Evaluation

Models are evaluated using standard classification metrics:

- Accuracy
- F1-score
- ROC-AUC

These metrics provide insight into model performance, particularly for handling **imbalanced churn data**.

---

## Quick Start

### Prerequisites

- Python 3.8+
- 4GB+ RAM
- Jupyter Notebook or JupyterLab

---

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/churn-prediction.git

cd churn-prediction

# Create virtual environment
python -m venv venv

# Activate environment

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in the following order:

1. `01_load_data.ipynb`
2. `02_preprocessing.ipynb`
3. `03_optuna_hyperparameter_tuning.ipynb`
4. `04_models_training.ipynb`

---





