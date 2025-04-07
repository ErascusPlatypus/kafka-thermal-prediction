from kafka import KafkaConsumer
from river import tree, compose, preprocessing, metrics, drift
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import os

# Create results directory
os.makedirs("results", exist_ok=True)

# Improved model with just the essentials
model = compose.Pipeline(
    # Standardize features for better convergence
    preprocessing.StandardScaler(),
    
    # Improved Hoeffding Tree with better parameters
    tree.HoeffdingTreeRegressor(
        grace_period=50,       # More frequent split attempts
        delta=0.01,            # Confidence level
        max_depth=15,          # Allow slightly deeper trees
        leaf_prediction='mean' # Simple but effective
    )
)

# Feature order
FEATURE_ORDER = [
    'weekday', 'Voltage (V)', 'Current (A)', 'Power (PA) - Watts (W)',
    'Frequency - Hertz (Hz)', 'Active Energy - kilowatts per hour (KWh)',
    'Power factor - Adimentional',
    'ESP32 temperature - Centigrade Degrees (°C)',
    'CPU consumption - Percentage (%)', 'CPU power consumption - Percentage (%)',
    'GPU consumption - Percentage (%)', 'GPU power consumption - Percentage (%)',
    'GPU temperature - Centigrade Degrees (°C)', 'RAM memory consumption - Percentage (%)',
    'RAM memory power consumption - Percentage (%)'
]
TARGET_COL = 'CPU temperature - Centigrade Degrees (°C)'

# Metrics
rmse = metrics.RMSE()
mae = metrics.MAE()
r2 = metrics.R2()

# Simple drift detector
drift_detector = drift.ADWIN()

# Simple arrays for tracking results
y_true = []
y_pred = []
drift_points = []

# Connect to Kafka
print("Starting Hoeffding Tree consumer for CPU temperature prediction")
consumer = KafkaConsumer(
    "sensor-data",
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Start processing
sample_count = 0
start_time = time.time()

try:
    for message in consumer:
        data = message.value
        
        # Convert features to dictionary format for River
        x = {col: float(data.get(col, 0)) for col in FEATURE_ORDER}
        
        # Get target
        if TARGET_COL not in data:
            print(f"Warning: Target column '{TARGET_COL}' not found in message")
            continue
            
        y = float(data[TARGET_COL])
        
        # Make prediction (returns None until grace period is met)
        pred = model.predict_one(x)
        
        # Handle None prediction for initial samples
        if pred is None:
            pred = y  # Use actual as fallback
            
        # Calculate error
        error = abs(y - pred)
        
        # Update drift detector
        drift_detector.update(error)
        if drift_detector.drift_detected:
            print(f"Concept drift detected at sample {sample_count}")
            drift_points.append(sample_count)
            drift_detector.reset()
        
        # Update metrics
        rmse.update(y, pred)
        mae.update(y, pred)
        r2.update(y, pred)
        
        # Store values
        y_true.append(y)
        y_pred.append(pred)
        
        # Update model
        model.learn_one(x, y)
        
        # Print progress
        sample_count += 1
        # print(f"Sample {sample_count}: Pred={pred:.2f}°C, Actual={y:.2f}°C, Error={error:.2f}°C")
        print('y_pred : ', pred, ' Actual temp : ', y, 'Current iteration : ', sample_count)

        # Print metrics periodically
        if sample_count % 100 == 0:
            print(f"\nMetrics after {sample_count} samples:")
            print(f"RMSE: {rmse.get():.4f}°C")
            print(f"MAE: {mae.get():.4f}°C")
            print(f"R²: {r2.get():.4f}")
            
            # Simple visualization
            plt.figure(figsize=(12, 6))
            
            # Show recent 100 samples
            recent = min(100, len(y_true))
            plt.plot(range(sample_count-recent, sample_count), y_true[-recent:], 'b-', label='Actual')
            plt.plot(range(sample_count-recent, sample_count), y_pred[-recent:], 'r-', label='Predicted')
            
            # Mark drift points
            for point in drift_points:
                if point > sample_count-recent:
                    plt.axvline(x=point, color='g', linestyle='--')
            
            plt.title(f'CPU Temperature: Actual vs Predicted (Recent {recent} samples)')
            plt.xlabel('Sample')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f'results/prediction_{sample_count}.png')
            plt.close()

except KeyboardInterrupt:
    print("\nKafka consumer stopped by user")

finally:
    # Print final results
    if sample_count > 0:
        print("\nFinal Results:")
        print(f"Samples processed: {sample_count}")
        print(f"Total time: {time.time() - start_time:.2f} seconds")
        print(f"RMSE: {rmse.get():.4f}°C")
        print(f"MAE: {mae.get():.4f}°C")
        print(f"R²: {r2.get():.4f}")
        
        # Create final visualization
        plt.figure(figsize=(10, 6))
        plt.plot(y_true, 'b-', label='Actual', alpha=0.7)
        plt.plot(y_pred, 'r-', label='Predicted', alpha=0.7)
        
        # Mark drift points
        for point in drift_points:
            plt.axvline(x=point, color='g', linestyle='--', alpha=0.7, label='Drift' if point == drift_points[0] else '')
        
        plt.title(f'CPU Temperature Prediction with Hoeffding Tree ({sample_count} samples)')
        plt.xlabel('Sample')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/final_results.png')
        plt.close()
        
        # Save simple summary
        with open('results/summary.txt', 'w') as f:
            f.write("Hoeffding Tree Online Learning Results\n")
            f.write("="*40 + "\n\n")
            f.write(f"Samples processed: {sample_count}\n")
            f.write(f"Processing time: {time.time() - start_time:.2f} seconds\n\n")
            f.write("Performance metrics:\n")
            f.write(f"RMSE: {rmse.get():.4f}°C\n")
            f.write(f"MAE: {mae.get():.4f}°C\n")
            f.write(f"R²: {r2.get():.4f}\n\n")
            if drift_points:
                f.write(f"Concept drift detected at samples: {', '.join(map(str, drift_points))}\n")
            else:
                f.write("No concept drift detected\n")