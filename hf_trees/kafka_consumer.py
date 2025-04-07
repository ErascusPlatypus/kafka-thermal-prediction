from kafka import KafkaConsumer
from river import tree, compose, preprocessing, metrics, drift
import json
import time

scaler = preprocessing.StandardScaler()

model = compose.Pipeline(
    scaler,
    tree.HoeffdingTreeRegressor(
        grace_period=100,       
        delta=0.01,             
        max_depth=10,           
    )
)

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

metric = metrics.RMSE()
predictions = []
actuals = []

start_time = time.time()
cnt = 0 

for msg in consumer:
    data = msg.value
    
    x = {col: float(data[col]) for col in FEATURE_ORDER}
    y = float(data[TARGET_COL])
    
    y_pred = model.predict_one(x)
    
    model.learn_one(x, y)
    
    metric.update(y, y_pred)
    predictions.append(y_pred)
    actuals.append(y)

    print('y_pred : ', y_pred, ' Actual temp : ', y, 'Current iteration : ', cnt)
    cnt += 1
    
    if len(predictions) % 100 == 0:
        print(f"Processed {len(predictions)} samples")
        print(f"Current RMSE: {metric.get():.2f}°C")
        print(f"Throughput: {len(predictions)/(time.time()-start_time):.1f} samples/sec")