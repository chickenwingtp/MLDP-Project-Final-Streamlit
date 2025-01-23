import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.write(
    """
# Car Fuel Consumption Prediction App
This app predicts the **fuel consumption (highway miles per gallon)** based on car features!
"""
)

model_pipeline = joblib.load("fuelconsumption.pkl")

st.sidebar.header("User Input Parameters")

def user_input_features():
    df = pd.read_csv('CarFuel_Dataset_Assignment.csv')

    drop_useless = ['car_ID', 'symboling', 'CarName', 'doornumber', 'carbody', 'enginelocation', 
                    'fuelsystem', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'stroke', 
                    'compressionratio', 'boreratio', 'price']
    df = df.drop(columns=drop_useless)

    categorical_columns = df.select_dtypes(include=['object']).columns

    encoded_df = pd.get_dummies(df, columns=categorical_columns)

    curbweight = st.sidebar.slider("Curb Weight (lbs)", 1500, 4000, 2500)
    enginesize = st.sidebar.slider("Engine Size (cubic inches)", 60, 320, 120)
    horsepower = st.sidebar.slider("Horsepower", 48, 288, 100)
    citympg = st.sidebar.slider("City MPG", 13, 49, 25)

    aspiration = st.sidebar.selectbox("Aspiration", ["std", "turbo"])
    cylindernumber = st.sidebar.selectbox(
        "Number of Cylinders",
        ["four", "six", "five", "eight", "two", "three", "twelve"],
    )
    drivewheel = st.sidebar.selectbox("Drive Wheel", ["fwd", "rwd", "4wd"])
    enginetype = st.sidebar.selectbox(
        "Engine Type", ["dohc", "ohcv", "ohc", "l", "rotor", "ohcf", "dohcv"]
    )
    fueltype = st.sidebar.selectbox("Fuel Type", ["gas", "diesel"])

    peakrpm = st.sidebar.slider("Peak RPM", 4150, 6600, 5000)

    data = {
        "fueltype": fueltype,
        "aspiration": aspiration,
        "drivewheel": drivewheel,
        "curbweight": curbweight,
        "enginetype": enginetype,
        "cylindernumber": cylindernumber,
        "enginesize": enginesize,
        "horsepower": horsepower,
        "peakrpm": peakrpm,
        "citympg": citympg,
    }

    input_df = pd.DataFrame(data, index=[0])

    input_df_encoded = pd.get_dummies(input_df, columns=categorical_columns)
    input_df_encoded = input_df_encoded.reindex(columns=encoded_df.columns, fill_value=0)
    input_df_encoded = input_df_encoded.drop('highwaympg', axis=1)

    return input_df_encoded


df = user_input_features()

st.subheader("User Input Parameters")
st.write(df)

try:
    prediction = model_pipeline.predict(df)

    st.subheader("Predicted Fuel Consumption (Highway MPG)")
    st.write(
        f"The estimated fuel consumption (highway mpg) for the selected car is: **{prediction[0]:.2f} MPG**"
    )
except Exception as e:
    st.error(f"Error making prediction: {str(e)}")
