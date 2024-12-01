## Emotion Detection In Text

### **Overview**
This project is a web application for detecting emotions in text using machine learning. The application is built with **Streamlit** for the frontend and **Scikit-learn** for the backend model. It allows users to input text and receive a predicted emotion along with the confidence level.

# Requirements
### Python Libraries:
Install the required libraries using pip:
```
pip install -r requirements.txt
```
### **Dependencies**

-   `streamlit`
-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `joblib`
-   `altair`

# Project Structure
```
Emotion-Detection
├── app.py      // Entry start point
├── data
│   └── emotion_dataset.csv      // Training dataset
├── lib
│   ├── main.py      // Streamlit application
│   ├── model_operation.py      // Handles model predictions
│   └── train_model.py      // Script to train and save the ML model
└── requirements.txt      // Project documentation (this file)
```

# Setup and Usage
**1. Train Model**
Run the training script to create the model and vectorizer:
```
python train_model.py
```
\
**2. Run the Streamlit App**
Launch the Streamlit web application:
```
streamlit run app.py
```
Open your browser and go to `http://localhost:8501` to interact with the app.

# How It Works
1.  **Training**: The `train_model.py` script uses Scikit-learn to train a logistic regression model on the dataset.
2.   **Model Prediction**: The `model_operation.py` file provides functions for predicting emotions and their probabilities using the saved model.
3. **Streamlit App**: The frontend allows users to input text, see the emotion prediction, and visualize probabilities.

# Example
![Screenshot from 2024-12-01 11-50-41](https://github.com/user-attachments/assets/3cc3d895-4052-4295-bcbf-22b8dc0407eb)
