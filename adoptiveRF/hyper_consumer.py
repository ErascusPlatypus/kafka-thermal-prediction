from kafka import KafkaConsumer
from river import ensemble, forest, compose, preprocessing, metrics, drift, utils
import json
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
import numpy as np
import os

# Create results directory
os.makedirs("results/tuning", exist_ok=True)

# ======================
# HYPERPARAMETER CONFIGURATIONS
# ======================
@dataclass
class ModelConfig:
    n_models: int
    max_depth: int
    lambda_value: int
    grace_period: int
    aggregation_method: str
    disable_weighted_vote: bool

configurations = [
    ModelConfig(15, 20, 8, 50, 'mean', False),      # Aggressive learning
    ModelConfig(20, 25, 10, 75, 'median', True),    # Conservative approach
    ModelConfig(12, 18, 6, 40, 'mean', False),      # Balanced configuration
    ModelConfig(25, 30, 12, 100, 'median', True)    # Large ensemble
]

# ======================
# TUNER CLASS
# ======================
class HyperparameterTuner:
    def __init__(self, configs):
        self.models = []
        self.metrics = []
        self.best_model_idx = 0
        self.sample_window = 200  # Samples between evaluations
        
        # Initialize models with different configurations
        for config in configs:
            model = compose.Pipeline(
                preprocessing.StandardScaler(),
                forest.ARFRegressor(
                    n_models=config.n_models,
                    max_depth=config.max_depth,
                    lambda_value=config.lambda_value,
                    grace_period=config.grace_period,
                    aggregation_method=config.aggregation_method,
                    disable_weighted_vote=config.disable_weighted_vote,
                    drift_detector=drift.ADWIN(0.001),
                    warning_detector=drift.ADWIN(0.01),
                    metric=metrics.RMSE(),
                    seed=42
                )
            )
            self.models.append({
                'config': config,
                'instance': model,
                'rmse': utils.Rolling(metrics.RMSE(), window_size=100),
                'mae': utils.Rolling(metrics.MAE(), window_size=100),
                'history': []
            })

    def update_models(self, x, y):
        """Update all models with new data point"""
        for model in self.models:
            pred = model['instance'].predict_one(x)
            if pred is None:
                pred = y  # Initial fallback
            
            # Update model and metrics
            model['instance'].learn_one(x, y)
            model['rmse'].update(y, pred)
            model['mae'].update(y, pred)
            model['history'].append((y, pred))

    def evaluate_models(self, sample_count):
        """Select best performing model based on rolling RMSE"""
        best_performance = float('inf')
        for idx, model in enumerate(self.models):
            current_rmse = model['rmse'].get()
            if current_rmse < best_performance:
                best_performance = current_rmse
                self.best_model_idx = idx

        # Log evaluation
        print(f"\n=== Evaluation at sample {sample_count} ===")
        for idx, model in enumerate(self.models):
            print(f"Model {idx} [{model['config']}]")
            print(f"Rolling RMSE: {model['rmse'].get():.2f}°C")
            print(f"Rolling MAE: {model['mae'].get():.2f}°C")
            print("---")

    def get_best_predictor(self):
        return self.models[self.best_model_idx]

# ======================
# MAIN PROCESSING
# ======================
tuner = HyperparameterTuner(configurations)
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

# Kafka consumer setup
consumer = KafkaConsumer(
    "sensor-data",
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Tracking variables
sample_count = 0
start_time = time.time()
performance_history = []

try:
    for message in consumer:
        data = message.value
        x = {col: float(data.get(col, 0)) for col in FEATURE_ORDER}
        y = float(data[TARGET_COL])
        
        # Update all models
        tuner.update_models(x, y)
        sample_count += 1
        
        # Periodic evaluation and logging
        if sample_count % tuner.sample_window == 0:
            tuner.evaluate_models(sample_count)
            performance_history.append({
                'sample': sample_count,
                'models': [m['rmse'].get() for m in tuner.models]
            })
            
            # Plot performance comparison
            plt.figure(figsize=(12, 6))
            for idx in range(len(tuner.models)):
                model_rmses = [ph['models'][idx] for ph in performance_history]
                samples = [ph['sample'] for ph in performance_history]
                plt.plot(samples, model_rmses, 
                        label=f"Model {idx}", 
                        linestyle='--' if idx != tuner.best_model_idx else '-',
                        linewidth=2 if idx == tuner.best_model_idx else 1)
            
            plt.title("Hyperparameter Tuning Progress")
            plt.ylabel("Rolling RMSE (°C)")
            plt.xlabel("Samples Processed")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"results/tuning/performance_{sample_count}.png")
            plt.close()

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    # Generate final comparison report
    best_model = tuner.get_best_predictor()
    print("\n=== Final Tuning Results ===")
    print(f"Total samples processed: {sample_count}")
    print(f"Total time: {time.time() - start_time:.2f}s")
    print(f"\nBest model configuration: {best_model['config']}")
    print(f"Final Rolling RMSE: {best_model['rmse'].get():.2f}°C")
    print(f"Final Rolling MAE: {best_model['mae'].get():.2f}°C")

    # Plot all models' error trajectories
    plt.figure(figsize=(12, 6))
    for idx in range(len(tuner.models)):
        rmses = [ph['models'][idx] for ph in performance_history]
        samples = [ph['sample'] for ph in performance_history]
        plt.plot(samples, rmses, label=f"Model {idx}")

    plt.title("Hyperparameter Tuning Trajectories")
    plt.ylabel("Rolling RMSE (°C)")
    plt.xlabel("Samples Processed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/tuning/final_tuning_comparison.png")
    plt.close()