import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import plost
import pickle

MODEL_PATH = f'weather_model.pkl'
SCALER_PATH = f'scaler.pkl'

# Function to load the Model and the Scaler
def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

model = load_pkl(MODEL_PATH)
scaler = load_pkl(SCALER_PATH)

def get_clean_data():
    data = pd.read_csv("weather_dataset.csv")
    return data

def add_sidebar():
    st.sidebar.header("Weather Predictor `App ⛈️`")
    st.sidebar.write("This Artificial Intelligence App predicts the future weather based on input parameters.")

    st.sidebar.subheader('Select the Weather Parameters ✅:')

    data = get_clean_data()

    slider_labels = [
        ("Precipitation", "precipitation"),
        ("Max Temperature", "temp_max"),
        ("Min Temperature", "temp_min"),
        ("Wind", "wind"),
    ]
    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['weather'], axis=1)

    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Precipitation', 'Max Temperature', 'Min Temperature', 'Wind']

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['precipitation'], input_data['temp_max'], input_data['temp_min'],
            input_data['wind']
        ],
        theta=categories,
        fill='toself',
        name='Input Values'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )
    return fig

def add_predictions(input_data):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    pred_result = model.predict(input_array_scaled)

    pred_result = int(pred_result[0])
    probabilities = model.predict_proba(input_array_scaled)[0]
    weather_classes = ['Drizzle', 'Rain', 'Sun', 'Snow', 'Fog']
    
    st.markdown("### Weather Prediction ✅")
    st.write("<span class='diagnosis-label'>Machine Learning Model Result:</span>", unsafe_allow_html=True)
    st.write(f"<span class='diagnosis {weather_classes[pred_result].lower()}'>{weather_classes[pred_result]}</span>", unsafe_allow_html=True)
    
    cols = st.columns(2)
    for idx, weather in enumerate(weather_classes):
        with cols[idx % 2]:
            st.metric(f"{weather} Probability:", f"{probabilities[idx] * 100:.2f}%")

    st.write("`This AI app can assist with weather predictions but should not replace professional forecasting.`")

def main():
    st.set_page_config(
        page_title="Weather Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    input_data = add_sidebar()

    st.title("Weather Predictor ⛅️")
    st.write("This app predicts weather using a KNeighborsClassifier model. Adjust parameters using the sliders on the sidebar.")

    col1, col2 = st.columns([2, 1])
    df = pd.read_csv("weather_classes.csv")

    with col1:
        st.markdown('### Radar Chart of the Parameters 📊')
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

        st.markdown('### Bar Chart of the Weather Classes 📉')
        plost.bar_chart(
            data=df,
            bar='Weather',
            value='Number of that Class',
            legend='bottom',
            use_container_width=True,
            color='Weather'
        )

    with col2:
        st.markdown('### Donut Chart of the Weather Classes 📈')
        plost.donut_chart(
            data=df,
            theta="Number of that Class",
            color='Weather',
            legend='bottom',
            use_container_width=True
        )
        add_predictions(input_data)

if __name__ == "__main__":
    main()
