import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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
    st.write(f"), the prediction using {best_model_name} model is: {'Diabetes' if prediction > 0.5 else 'No Diabetes'}")

    if best_model_name == "Linear Regression":
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error for {model_name} model: {mse}")
        st.write(f"R^2 Score for {model_name} model: {r2}")

    else:    
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        st.write(f"Confusion Matrix for {best_model_name} model:")
        cm = confusion_matrix(y_test, best_model.predict(X_test))
        st.write(pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0', 'Actual 1']))
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        st.write(f"Precision for {model_name} model: {precision}")
        st.write(f"Recall for {model_name} model: {recall}")
        st.write(f"F1 Score for {model_name} model: {f1}")
        st.write(f"Accuracy for {model_name} model: {accuracy}")

    st.table(pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy']))

if __name__ == "__main__":
    main()

# Copyright text at the bottom
st.sidebar.markdown(
    '<div style="text-align:center; margin-top: 370px">'
    '<a href="https://pranavsuriya-sr.github.io/personalPortfolio/" style="text-decoration: none;" ><p style="font-size: 10px;">PS Devs © 2023 Project Hack Community.</a></p>'
    '<p style="font-size: 10px;">Open Source rights reserved.</p>'
    '</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div style="text-align:center; margin-top: 42px">'
    '<a href="https://pranavsuriya-sr.github.io/personalPortfolio/" style="text-decoration: none;" ><p style="font-size: 10px;">PS Devs © 2023 Project Hack Community.</a></p>'
    '<p style="font-size: 10px;">Open Source rights reserved.</p>'
    '</div>',
    unsafe_allow_html=True
)