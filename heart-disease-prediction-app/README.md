# Heart Disease Prediction App

A Streamlit app that predicts heart disease risk from patient details.

**Live app:** https://ml-beginner-projects-zqhrbuhaetkxmj6hhjnnbt.streamlit.app/

## About
The app takes 11 inputs such as age, resting BP, cholesterol, chest pain type and max heart rate. It scales the inputs with a saved scaler and predicts risk with a Logistic Regression model. It shows the risk percentage and a Low or High result.

## Model
- Dataset: heart.csv (918 rows, target: HeartDisease)
- Cleaning: zero values in Cholesterol and RestingBP replaced with the column mean
- Encoding: one-hot encoding for categorical columns
- Scaling: StandardScaler
- Models compared: Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM
- Final model: Logistic Regression (accuracy: XX%, F1 score: XX%)

## Tech used
Python, pandas, scikit-learn, joblib, Streamlit

## Files
- `app.py`: the Streamlit app
- `Logistic_Regression_model.pkl`: trained model
- `scaler.pkl`: fitted StandardScaler
- `columns.pkl`: column order used in training
- `heart.csv`: dataset
- `heart-disease-prediction.ipynb`: EDA, preprocessing and model training
- `requirements.txt`: packages needed

## Run locally
    pip install -r requirements.txt
    python -m streamlit run app.py

## Disclaimer
This is a student project built for learning. It is not medical advice. Please consult a doctor for any health concern.

Built by Sameer
