
# ML Beginner Projects

A collection of beginner machine learning projects built with Python, pandas, scikit-learn, seaborn and matplotlib. Each project is one notebook.

## Projects

| # | Project | Task | Notebook |
|---|---------|------|----------|
| 1 | Insurance Charges | Regression | 1.ipynb |
| 2 | Heart Disease | Classification | 2.ipynb |
| 3 | Titanic Survival | Classification | TITANIC_SURVIVAL.ipynb |
| 4 | Ford Car Prediction | Regression | ford.ipynb |

## 1. Insurance Charges
Predicts medical insurance charges from age, BMI, smoking status and other features.
Steps: EDA, data cleaning, feature check with Pearson correlation and p-values, model training.

## 2. Heart Disease
Predicts whether a patient has heart disease from clinical features.
Steps: EDA, preprocessing, model training and evaluation.
Result: 79% accuracy, 83% F1.

## 3. Titanic Survival
Predicts which passengers survived. Five models compared on the same 80/20 split.

| Model | Accuracy | Survivor F1 |
|-------|----------|-------------|
| SVM (RBF) | 81% | 0.77 |
| Logistic Regression | 80% | 0.75 |
| Decision Tree | 80% | 0.75 |
| KNN | 79% | 0.74 |
| Naive Bayes | 78% | 0.73 |

Steps: missing value handling, label encoding, StandardScaler, model comparison.

## 4. Ford Car Prediction
Predicts the price of a used Ford car from features such as year, mileage, fuel type, transmission and engine size.
Steps: EDA, data cleaning, encoding, model training and evaluation.

## Tools
Python, NumPy, pandas, seaborn, matplotlib, scikit-learn

## How to run
1. Clone the repo
2. Install the libraries: `pip install numpy pandas seaborn matplotlib scikit-learn`
3. Open any notebook in Jupyter and run all cells

## Author
Sameer Syed
GitHub: [SAMEER-SYED114](https://github.com/SAMEER-SYED114)
