
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model, label encoders, and feature columns
model = joblib.load('best_traffic_volume_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Streamlit app title
st.title('Prediksi Volume Lalu Lintas')
st.write('Aplikasi ini memprediksi volume lalu lintas berdasarkan kondisi cuaca, waktu, dan hari libyur.')

# Input fields for user
# Categorical features
holiday_options = label_encoders['holiday'].classes_
weather_main_options = label_encoders['weather_main'].classes_
weather_description_options = label_encoders['weather_description'].classes_

holiday = st.selectbox('Holiday', holiday_options)
weather_main = st.selectbox('Weather Main', weather_main_options)
weather_description = st.selectbox('Weather Description', weather_description_options)

# Numerical features
temp = st.slider('Temperature (Kelvin)', min_value=250.0, max_value=320.0, value=290.0, step=0.1)
rain_1h = st.number_input('Rain in last hour (mm)', min_value=0.0, max_value=100.0, value=0.0, step=0.01)
snow_1h = st.number_input('Snow in last hour (mm)', min_value=0.0, max_value=100.0, value=0.0, step=0.01)
clouds_all = st.slider('Clouds all (%)', min_value=0, max_value=100, value=50, step=1)
hour = st.slider('Hour of Day (0-23)', min_value=0, max_value=23, value=12, step=1)
day = st.slider('Day of Month (1-31)', min_value=1, max_value=31, value=15, step=1)
month = st.slider('Month (1-12)', min_value=1, max_value=12, value=7, step=1)
day_of_week = st.slider('Day of Week (0=Monday, 6=Sunday)', min_value=0, max_value=6, value=2, step=1)

# Create a button to make prediction
if st.button('Predict Traffic Volume'):
    # Encode categorical inputs
    encoded_holiday = label_encoders['holiday'].transform([holiday])[0]
    encoded_weather_main = label_encoders['weather_main'].transform([weather_main])[0]
    encoded_weather_description = label_encoders['weather_description'].transform([weather_description])[0]

    # Create a DataFrame for the input
    input_data = pd.DataFrame([[encoded_holiday, temp, rain_1h, snow_1h, clouds_all,
                                  encoded_weather_main, encoded_weather_description,
                                  hour, day, month, day_of_week]],
                                columns=feature_columns)

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.success(f'Predicted Traffic Volume: {int(prediction):,}')

st.markdown("""
**How to run this app:**
1. Save the model, encoders, and feature columns using the cells above.
2. In your terminal, navigate to the directory where `streamlit_app.py` is saved.
3. Run `streamlit run streamlit_app.py`
""")
