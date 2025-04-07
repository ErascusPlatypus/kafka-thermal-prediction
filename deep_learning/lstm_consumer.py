import json
import torch
import torch.nn as nn
from kafka import KafkaConsumer
from deep_river import regression
from river import preprocessing, metrics

# --- Configuration ---
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
INPUT_FEATURES = [f for f in FEATURE_ORDER if f != TARGET_COL] # Use all other features as input
N_FEATURES = len(INPUT_FEATURES)

KAFKA_TOPIC = "sensor-data"
KAFKA_BROKER = 'localhost:9092' # Assumes Kafka is running locally

# Model Hyperparameters
WINDOW_SIZE = 100        # Increased window to capture longer patterns
HIDDEN_SIZE = 128       # Increased capacity for complex relationships
NUM_LSTM_LAYERS = 2     # Deeper network with dropout
DROPOUT = 0.3           # Regularization to prevent overfitting
LEARNING_RATE = 0.0005  # Lower learning rate for stability
LOSS_FN = 'smooth_l1'   # Huber loss (robust to outliers)
OPTIMIZER_FN = 'adam'   
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

print(f"Using device: {DEVICE}")
torch.manual_seed(SEED)
if DEVICE == 'cuda':
    torch.cuda.manual_seed(SEED)

class Predictor(nn.Module):
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,  
            batch_first=False
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, hidden_size//2)
        self.linear2 = nn.Linear(hidden_size//2, 1)
        self.activation = nn.LeakyReLU(0.01)  #

    def forward(self, x, **kwargs):
        out, _ = self.lstm(x)
        final_out = out[-1, :]        # Take last timestep
        final_out = self.dropout(final_out)
        final_out = self.activation(self.linear(final_out))
        return self.linear2(final_out)

lstm_module = Predictor(
    n_features=N_FEATURES,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LSTM_LAYERS,
    dropout=DROPOUT
)
model_pipeline = regression.RollingRegressorInitialized(
    module=lstm_module,
    loss_fn=LOSS_FN,
    optimizer_fn=OPTIMIZER_FN, 
    lr=LEARNING_RATE,
    window_size=WINDOW_SIZE,
    device=DEVICE,
    seed=SEED,
    append_predict=False       

)


model_pipeline = preprocessing.StandardScaler() | model_pipeline

metric = metrics.RMSE() # Root Mean Squared Error

consumer = KafkaConsumer(
    KAFKA_TOPIC,
    bootstrap_servers=[KAFKA_BROKER],
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    auto_offset_reset='latest', 
    consumer_timeout_ms=10000 
)

print(f"Subscribed to Kafka topic: {KAFKA_TOPIC} on {KAFKA_BROKER}")
print("Starting online learning loop...")

count = 0
try:
    for message in consumer:
        count += 1
        data = message.value 
        x = {col: float(data[col]) for col in FEATURE_ORDER}
        y = float(data[TARGET_COL])

        pred = model_pipeline.predict_one(x)

        model_pipeline.learn_one(x=x, y=y)
        metric.update(y_true=y, y_pred=pred)

        print(f'Predicted Temp: {pred:.2f}, Actual Temp: {y:.2f}, Iteration: {count}')


        if count % 100 == 0:
            if pred is not None: # Only print metrics if we have predictions
                print(f"Processed: {count} | Current RMSE : {metric.get():.2f}°C")
            else:
                print(f"Processed: {count} | Filling initial window...")


except KeyboardInterrupt:
    print("Interrupted by user.")
except Exception as e:
    print(f"An error occurred in the consumer loop: {e}")
finally:
    print("Closing Kafka consumer...")
    consumer.close()
    print("Final evaluation:")
    print(metric)

print("Online learning finished.")