import streamlit as st
import pandas as pd
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

@st.cache
def load_data():
    # Load the dataset and drop the "Initial release" and "game" columns
    df = pd.read_csv("mobile.csv")
    df = df.drop(columns=["Initial release", "Game"])
    return df

def load_and_predict(df):
    # Split the data into training and testing sets
    X_train = df[["Publisher(s)", "Genre(s)"]]
    y_train = df["Revenue"]
    X_test = X_train
    y_test = y_train

    # One-hot encode the categorical variables
    encoder = OneHotEncoder()
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    # Train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train_encoded, y_train)

    # Get user input for Publisher(s) and Genre(s)
    publisher = st.selectbox("Publisher(s)", df["Publisher(s)"].unique())
    genre = st.selectbox("Genre(s)", df["Genre(s)"].unique())
    input_data = pd.DataFrame([[publisher, genre]], columns=X_train.columns)

    # One-hot encode the user input
    input_data_encoded = encoder.transform(input_data)

    # Disable the warning message
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but LinearRegression was fitted with feature names")

    # Make predictions using the user input
    y_pred = model.predict(input_data_encoded)

    # Display the prediction to the user
    st.markdown(
        """
        <style>
            h1 {
                color: blue;
                text-align: center;
            }
            p {
                font-size: 20px;
                font-weight: bold;
            }
        </style>
        <h1>Video Game Revenue Calculator</h1>
        <p>Enter the publisher and genre of the video game to predict its revenue:</p>
        """
    , unsafe_allow_html=True)
    st.write("Predicted revenue (Millions of USD): ", y_pred[0])

def main():
    df = load_data()
    load_and_predict(df)

if __name__ == "__main__":
    main()