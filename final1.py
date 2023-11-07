import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

def main():
    st.markdown("# Diabetes Predictor App")
    st.markdown("This predictor utilizes advanced algorithms to assess your risk of developing diabetes based on your input and known risk factors.")

    st.sidebar.markdown("### Parameter Selection: ")
    selected_option = st.sidebar.selectbox('Select the number of parameters you know: ', ('1', '2', '3', '4', '5', '6', '7', '8'))

    feature_names = {
        'Pregnancies': 'Pregnancies',
        'Glucose': 'Glucose',
        'BloodPressure': 'BloodPressure',
        'SkinThickness': 'SkinThickness',
        'Insulin': 'Insulin',
        'BMI': 'BMI',
        'DiabetesPedigreeFunction': 'DiabetesPedigreeFunction',
        'Age': 'Age'
    }

    selected_features = [st.sidebar.selectbox(f'Select parameter {i}:', list(feature_names.values())) for i in range(1, int(selected_option) + 1)]

    numbers = {}
    for i, value in enumerate(selected_features, start=1):
        number = st.number_input(f'Enter value for {value}', key=str(i))
        numbers[value] = number

    df = pd.read_csv('diabetes.csv')  

    # Selecting columns based on user input
    selected_columns = [key for key in numbers.keys()]

    # Preparing the feature matrix and target variable
    X = df[selected_columns]
    y = df['Outcome']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
            'Linear Regression': LinearRegression(),
            'Logistic Regression': LogisticRegression(),
            'KNN': KNeighborsClassifier(),
            'Random Forest': RandomForestClassifier(),
            'SVM Linear': SVC(kernel='linear'),
            'SVM RBF': SVC(kernel='rbf'),
            'SVM Polynomial': SVC(kernel='poly'),
            'SVM Sigmoid': SVC(kernel='sigmoid')
        }

    accuracies = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)  # Calculate accuracy on the test set
        accuracies[model_name] = accuracy


    # Finding the model with the highest accuracy
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]

    # Train the best model on the entire dataset
    best_model.fit(X, y)

    # Making predictions based on user inputs
    user_input = pd.DataFrame(numbers, index=[0])
    prediction = best_model.predict(user_input)

    st.write(f"Based on the input (")
    for key, value in numbers.items():
        st.write(f"{key}: {value},")
    st.write(f"), the prediction using {best_model_name} model is: {'Diabetes' if prediction == 1 else 'No Diabetes'}")


    st.table(pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy']))


if __name__ == "__main__":
    main()

# Copyright text at the bottom
st.sidebar.markdown(
    '<div style="text-align:center; margin-top: 370px">'
    '<a href = "https://pranavsuriya-sr.github.io/personalPortfolio/" style = "text-decoration: none;" ><p style="font-size: 10px;">PS Devs © 2023 Project Hack Community.</a></p>'
    '<p style="font-size: 10px;">Open Source rights reserved.</p>'
    '</div>',
    unsafe_allow_html=True
)
st.markdown(
        '<div style="text-align:center; margin-top: 42px">'
        '<a href = "https://pranavsuriya-sr.github.io/personalPortfolio/" style = "text-decoration: none;" ><p style="font-size: 10px;">PS Devs © 2023 Project Hack Community.</a></p>'
        '<p style="font-size: 10px;">Open Source rights reserved.</p>'
        '</div>',
        unsafe_allow_html=True
    )
