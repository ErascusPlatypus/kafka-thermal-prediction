from kafka import KafkaConsumer
from joblib import load
from sklearn.metrics import mean_squared_error
import numpy as np
import json
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import os  # Added to handle directory creation

warnings.filterwarnings("ignore")

# Ensure that the results directory exists
os.makedirs("results", exist_ok=True)

model = load("paper_model.joblib")
y_true = []
y_pred = []
rmse_history = []
iterations = []  

FEATURE_ORDER = [
    'weekday', 'Voltage (V)', 'Current (A)', 'Power (PA) - Watts (W)',
    'Frequency - Hertz (Hz)', 'Active Energy - kilowatts per hour (KWh)',
    'Power factor - Adimentional',
    'ESP32 temperature - Centigrade Degrees (°C)',
    'CPU consumption - Percentage (%)',
    'CPU power consumption - Percentage (%)',
    'GPU consumption - Percentage (%)',
    'GPU power consumption - Percentage (%)',
    'GPU temperature - Centigrade Degrees (°C)',
    'RAM memory consumption - Percentage (%)',
    'RAM memory power consumption - Percentage (%)'
]
TARGET_COL = 'CPU temperature - Centigrade Degrees (°C)'

consumer = KafkaConsumer(
    "sensor-data",
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

cnt = 0 
try:
    for message in consumer:
        data = message.value
        
        features_dict = {col: data.get(col, 0) for col in FEATURE_ORDER}
        features_df = pd.DataFrame([features_dict])
        
        true_temp = data[TARGET_COL]

        pred_temp = model.predict(features_df)[0]
        
        y_true.append(true_temp)
        y_pred.append(pred_temp)
        
        print('y_pred : ', pred_temp, ' Actual temp : ', true_temp, 'Current iteration : ', cnt)
        cnt += 1
        
        model.partial_fit(features_df, [true_temp])
        
        if len(y_true) % 100 == 0:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            print(f"RMSE after {len(y_true)} samples: {rmse:.2f}°C")
            rmse_history.append(rmse)
            iterations.append(len(y_true))
except KeyboardInterrupt:
    print("Consumer stopped by user.")

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rmse_history, marker='o', linestyle='-')
    plt.xlabel("Samples")
    plt.ylabel("RMSE (°C)")
    plt.title("RMSE over Time")
    plt.grid(True)
    plt.savefig("results/clean_rmse_over_time.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rmse_history, marker='o', linestyle='-')
    plt.xlabel("Samples")
    plt.ylabel("RMSE (°C)")
    plt.yscale('log')
    plt.title("RMSE over Time")
    plt.grid(True)
    plt.savefig("results/log_rmse_over_time.png")
    plt.close()
