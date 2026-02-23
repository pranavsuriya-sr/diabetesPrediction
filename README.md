# Diabetes Predictor App

A machine learning-powered web application built with Streamlit to assess the risk of developing diabetes based on user input and known risk factors.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models Evaluated](#models-evaluated)
- [License & Credits](#license--credits)

## Overview
This predictor utilizes advanced machine learning algorithms to assist in evaluating diabetes risk. It dynamically trains and evaluates multiple models based on the selected dataset features (risk factors), picks the best-performing one, and provides a customized prediction alongside model evaluation metrics such as Accuracy, Precision, Recall, and F1 Score.

## Features
- **Dynamic Parameter Selection**: Users can choose how many risk factors they know (from 1 to 8) and input their specific values.
- **Multiple ML Models**: The application trains several models, including Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), various Support Vector Machines (SVM), and Linear Regression.
- **Auto-Model Selection**: Automatically finds the model with the highest test accuracy based on the user-selected features.
- **Performance Metrics**: Displays detailed evaluation metrics for the best model, including Confusion Matrix, Precision, Recall, F1 Score, and Accuracy.
- **Interactive UI**: Built with Streamlit for a responsive, wide-layout user interface.

## Technologies Used
- **Python 3**
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Scikit-Learn**: Machine learning models and metrics
- **Matplotlib & Seaborn**: Data visualization

## Installation

1. Clone the repository or navigate to the project directory.
2. Ensure you have Python installed on your system.
3. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Usage

1. Open a terminal or command prompt.
2. Navigate to the project directory where `final1.py` is located.
3. Run the Streamlit application with the following command:

```bash
streamlit run final1.py
```

4. Open the provided Local URL (usually `http://localhost:8501`) in your web browser.
5. From the sidebar, select the number of parameters you have data for.
6. Choose the specific parameters (e.g., Glucose, BMI, Age, BloodPressure, etc.) and enter their values.
7. The app will process your input, select the best model, make a prediction, and display whether there is a risk of diabetes.

## Dataset
The application uses the local `diabetes.csv` dataset, which contains various diagnostic measurements and an `Outcome` column indicating whether the patient has diabetes. The features include:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

## Models Evaluated
The following algorithms are trained and compared dynamically to provide your prediction:
- Logistic Regression
- Linear Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Support Vector Machines (SVM - Linear, RBF, Polynomial, Sigmoid kernels)

## License & Credits
- **Developer**: PS Devs
- © 2023 Project Hack Community
- Open Source rights reserved.
- Developer Portfolio: [pranavsuriya.netlify.app](https://pranavsuriya.netlify.app/)
