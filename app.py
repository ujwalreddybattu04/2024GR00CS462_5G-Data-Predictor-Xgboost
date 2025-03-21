import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background-color: #00CC96;
        color: white;
        border-radius: 10px;
        font-weight: bold;
    }
    .stSlider>div>div>div {
        color: #00CC96;
    }
    .stSelectbox>div>div>select {
        background-color: #2E2E2E;
        color: #FFFFFF;
    }
    .stExpander {
        background-color: #2E2E2E;
        border-radius: 10px;
    }
    .stSuccess {
        background-color: #1A3C34;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model with error handling
try:
    with open("xgboost_model.pkl", "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file 'xgboost_model.pkl' not found. Please ensure the file exists in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading the model: {str(e)}")
    st.stop()

# Load the dataset for data insights (optional, if available)
try:
    df = pd.read_csv("https://raw.githubusercontent.com/your-username/5g-data-predictor-xgboost/main/data/5G_Data_Usage_Prediction_Enhanced.csv")
except FileNotFoundError:
    df = None
    st.warning("⚠️ Dataset not found. Some data insights features will be disabled.")

# Define category mappings
device_types = {0: "Smartphone", 1: "Tablet", 2: "Laptop", 3: "Smartwatch", 4: "IoT Device"}
app_categories = {0: "Social Media", 1: "Video Streaming", 2: "Gaming", 3: "Productivity", 4: "Messaging", 5: "E-Commerce", 6: "Other"}
network_types = {0: "2G", 1: "3G", 2: "4G", 3: "5G"}
data_quality = {0: "Low", 1: "Medium", 2: "High"}
location_types = {0: "Urban", 1: "Suburban", 2: "Rural"}
indoor_outdoor = {0: "Indoor", 1: "Outdoor"}

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state['prediction_history'] = []

# Streamlit UI
st.title("📡 5G Data Usage Prediction App")
st.markdown("**Predict data usage with advanced machine learning models and gain insights into your 5G network usage.**")

# Theme Toggle
theme = st.sidebar.selectbox("🎨 Theme", ["Dark", "Light"])
if theme == "Light":
    st.markdown("""
        <style>
        .main { background-color: #FFFFFF; color: #000000; }
        .stSelectbox>div>div>select { background-color: #F0F0F0; color: #000000; }
        .stExpander { background-color: #F0F0F0; }
        </style>
    """, unsafe_allow_html=True)

# Sidebar for Model Insights
with st.sidebar:
    st.header("🛠️ Model Insights")
    with st.expander("Model Performance"):
        # Placeholder for model performance metrics (you can compute these if you have test data)
        st.markdown("""
            - **RMSE:** 45.32 MB  
            - **MAE:** 32.15 MB  
            - **R² Score:** 0.8923  
        """)
        st.info("These metrics are placeholders. Compute them using your test set for accurate results.")

    with st.expander("Feature Importance"):
        # Placeholder for feature importance (you can extract this from the XGBoost model)
        importance_df = pd.DataFrame({
            'Feature': ['session_duration', 'app_category', 'background_data', 'throughput', 'device_type'],
            'Importance': [0.35, 0.25, 0.15, 0.12, 0.08]
        })
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            title='Feature Importance (XGBoost)',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)

# Main Layout with Tabs
tab1, tab2, tab3 = st.tabs(["📊 Predict", "📈 Data Insights", "📜 Prediction History"])

# Tab 1: Prediction Interface
with tab1:
    st.subheader("Enter the required values for prediction:")
    
    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)
    
    with col1:
        selected_device_type = st.selectbox("📱 Device Type", list(device_types.values()))
        selected_app_category = st.selectbox("📂 App Category", list(app_categories.values()))
        session_duration = st.slider("⏳ Session Duration (mins)", 1, 300, 30)
        selected_data_quality = st.selectbox("📊 Data Quality", list(data_quality.values()))
        time_of_day = st.slider("🕒 Time of Day (0-23)", 0, 23, 12)
        day_of_week = st.slider("📅 Day of the Week (0=Monday, 6=Sunday)", 0, 6, 0)
    
    with col2:
        selected_network_type = st.selectbox("🌐 Network Type", list(network_types.values()))
        signal_strength = st.slider("📡 Signal Strength (dBm)", -120, -40, -80)
        prev_usage = st.slider("📂 Previous Usage (GB)", 0.0, 50.0, 5.0)
        selected_location_type = st.selectbox("📍 Location Type", list(location_types.values()))
        selected_indoor_outdoor = st.selectbox("🏠 Indoor/Outdoor", list(indoor_outdoor.values()))
        throughput = st.slider("🚀 Throughput (Mbps)", 0.0, 100.0, 10.0)
        background_data = st.slider("📊 Background Data (GB)", 0.0, 5.0, 1.0)

    # Convert selected options to numerical values
    try:
        device_type_num = list(device_types.keys())[list(device_types.values()).index(selected_device_type)]
        app_category_num = list(app_categories.keys())[list(app_categories.values()).index(selected_app_category)]
        network_type_num = list(network_types.keys())[list(network_types.values()).index(selected_network_type)]
        data_quality_num = list(data_quality.keys())[list(data_quality.values()).index(selected_data_quality)]
        location_type_num = list(location_types.keys())[list(location_types.values()).index(selected_location_type)]
        indoor_outdoor_num = list(indoor_outdoor.keys())[list(indoor_outdoor.values()).index(selected_indoor_outdoor)]
    except ValueError as e:
        st.error(f"❌ Invalid input selection: {str(e)}")
        st.stop()

    # Prepare input for model
    features = np.array([[device_type_num, app_category_num, session_duration, data_quality_num, time_of_day,
                          day_of_week, network_type_num, signal_strength, prev_usage, location_type_num,
                          indoor_outdoor_num, throughput, background_data]])

    # Predict
    if st.button("Predict Data Usage"):
        try:
            prediction = model.predict(features)
            st.success(f"📈 Predicted Data Usage: {prediction[0]:.2f} MB")

            # Save prediction to history
            prediction_entry = {
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Device Type': selected_device_type,
                'App Category': selected_app_category,
                'Session Duration (mins)': session_duration,
                'Data Quality': selected_data_quality,
                'Time of Day': time_of_day,
                'Day of Week': day_of_week,
                'Network Type': selected_network_type,
                'Signal Strength (dBm)': signal_strength,
                'Previous Usage (GB)': prev_usage,
                'Location Type': selected_location_type,
                'Indoor/Outdoor': selected_indoor_outdoor,
                'Throughput (Mbps)': throughput,
                'Background Data (GB)': background_data,
                'Predicted Data Usage (MB)': prediction[0]
            }
            st.session_state['prediction_history'].append(prediction_entry)

            # Visualize the prediction
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Predicted Data Usage'],
                y=[prediction[0]],
                marker_color='#00CC96',
                name='Predicted'
            ))
            fig.update_layout(
                title='Prediction Result',
                yaxis_title='Data Usage (MB)',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")

# Tab 2: Data Insights
with tab2:
    st.subheader("Data Insights")
    if df is not None:
        # Distribution of Data Usage by Device Type
        fig1 = px.box(
            df,
            x='device_type',
            y='data_usage',
            title='Data Usage Distribution by Device Type',
            labels={'device_type': 'Device Type', 'data_usage': 'Data Usage (MB)'},
            color='device_type',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Data Usage vs. Throughput
        fig2 = px.scatter(
            df,
            x='throughput',
            y='data_usage',
            color='network_type',
            title='Data Usage vs. Throughput by Network Type',
            labels={'throughput': 'Throughput (Mbps)', 'data_usage': 'Data Usage (MB)'},
            opacity=0.6
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Summary Statistics
        with st.expander("Summary Statistics"):
            st.write(df.describe())
    else:
        st.warning("⚠️ Dataset not available for insights.")

# Tab 3: Prediction History
with tab3:
    st.subheader("Prediction History")
    if st.session_state['prediction_history']:
        history_df = pd.DataFrame(st.session_state['prediction_history'])
        st.dataframe(history_df)

        # Download button for prediction history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Prediction History as CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

        # Visualize prediction trends over time
        fig = px.line(
            history_df,
            x='Timestamp',
            y='Predicted Data Usage (MB)',
            title='Prediction Trends Over Time',
            markers=True,
            color_discrete_sequence=['#00CC96']
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ No predictions made yet.")

# Footer
st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit | 5G Data Usage Prediction App**")
