📡 5G Data Usage Prediction App
🚀 Overview

The 5G Data Usage Prediction App is an AI-powered tool that predicts mobile data usage based on various factors like device type, network conditions, session duration, and more. The app leverages XGBoost for accurate predictions and provides an interactive interface using Streamlit.

📌 Features

✅ Predicts mobile data usage based on real-world parameters✅ Interactive UI built with Streamlit✅ Supports categorical and numerical inputs for accurate modeling✅ Uses XGBoost, a high-performance machine learning model✅ Live deployment capability with Streamlit Cloud

📂About dataset i have created my own dataet
![Screenshot 2025-03-21 025905](https://github.com/user-attachments/assets/516f1346-23c9-4cab-bcfe-8242f7fa382e)
Overview of My Dataset
My dataset contains information about 5G data usage based on different factors. Here’s a breakdown of the key columns:

1️⃣ User Information:
user_id → Unique identifier for each user.
2️⃣ Device & Application Usage:
device_type → Type of device (Smartphone, Laptop, Tablet, etc.).
app_category → Category of application (Gaming, Browsing, Video Streaming, etc.).
session_duration → Duration of the session in minutes.
data_quality → Quality of data used (Low, High, Ultra-HD, etc.).
3️⃣ Network & Connection Details:
time_of_day → Session time (Morning, Afternoon, Night, etc.).
day_of_week → Day when the session occurred.
network_type → Type of network (5G NSA, 5G SA, mmWave, Sub-6GHz, etc.).
signal_strength → Signal strength in dBm (Negative values indicate weaker signals).
prev_usage → Previous data usage in MB/GB.
4️⃣ Location & Environmental Factors:
location_type → Whether the user is in a Suburban, Urban, or Rural area.
indoor_outdoor → Indicates if the session was indoors or outdoors.
5️⃣ Performance & Data Consumption:
throughput → Network speed in Mbps.
background_usage → Background data consumption in GB.
data_usage → The actual data usage in MB/GB (Target variable for prediction

Technologies Used

Python (Machine Learning & Backend)

Streamlit (Frontend & Deployment)

XGBoost (Model Training)

Pandas & NumPy (Data Processing)

Plotly (Visualization)

web-app interface
![image](https://github.com/user-attachments/assets/d03e4ed1-5a1e-4008-941b-553d790ef114)

Git-clone:by uing thi link:
git clone https://github.com/ujwalreddybattu04/5G-Data-Predictor-Xgboost.git

🌐 Live Demo
The 5G Data Usage Prediction App is now live on Streamlit Cloud! 🚀
🔗 Access the app here: 👉 https://5g-data-predictor-xgboost-7gj2gkx36qj5y8rtwrxocc.streamlit.app/

The 5G Data Usage Prediction Model is built using XGBoost, a high-performance machine learning algorithm optimized for structured data. The model was trained and evaluated using multiple metrics to ensure high accuracy.
The model achieves exceptional accuracy, with an R² score of 0.9996, indicating near-perfect correlation between input factors and predicted data usage. The low MAE and RMSE further validate the model’s precision in forecasting mobile data consumption.
MAE:31.4670
RMSE:47.0258

References:
https://ieeexplore.ieee.org/document/10048885
https://www.sciencedirect.com/science/article/abs/pii/S1051200423004542

