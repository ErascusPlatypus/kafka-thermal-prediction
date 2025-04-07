from torch import nn 
from deep_river.regression import Regressor
from kafka import KafkaConsumer
import json
import numpy as np 
from river import preprocessing, metrics
import matplotlib.pyplot as plt

class Predictor(nn.Module):
    def __init__(self, n_features):
        super(Predictor, self).__init__()
        self.dense0 = nn.Linear(n_features, 128)
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.dense0(x))
        x = self.relu(self.dense1(x))
        x = self.relu(self.dense2(x))
        x = self.dense3(x) 
        return x 

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

model_pipeline = preprocessing.StandardScaler()
model_pipeline |= Regressor(module=Predictor, loss_fn="mse", optimizer_fn='sgd')

metric = metrics.RMSE()

consumer = KafkaConsumer(
    "sensor-data",
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

y_true = []
y_pred = []
rmse_history = []  
cnt = 0 

try:
    for message in consumer:
        data = message.value
        x = {col: float(data[col]) for col in FEATURE_ORDER}
        y = float(data[TARGET_COL])

        pred = model_pipeline.predict_one(x)
        metric.update(y_true=y, y_pred=pred)
        current_rmse = metric.get()
        rmse_history.append(current_rmse)
        
        y_true.append(y)
        y_pred.append(pred)

        print(f'Predicted Temp: {pred:.2f}, Actual Temp: {y:.2f}, Iteration: {cnt}')
        
        if len(y_pred) % 100 == 0:
            print(f"Processed {len(y_pred)} samples")
            print(f"Current RMSE: {current_rmse:.2f}°C")
            
        model_pipeline.learn_one(x, y)
        cnt += 1

except KeyboardInterrupt:
    print("Interrupted by user, stopping consumer...")

    plt.figure(figsize=(10, 6))
    plt.plot(rmse_history, label="RMSE")
    plt.xlabel("Samples")
    plt.ylabel("RMSE (°C)")
    plt.yscale('log')
    plt.title("RMSE over Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/log_rmse_results_over_time.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(rmse_history, label="RMSE")
    plt.xlabel("Samples")
    plt.ylabel("RMSE (°C)")
    plt.title("RMSE over Iterations")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/clean_rmse_results_over_time.png")
    plt.close()

    
