# train_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
import matplotlib.pyplot as plt

# Load data
print("Loading dataset...")
df = pd.read_csv("../../datasets/paper-datasets/train_data.csv", index_col=0)

# Define features and target
FEATURES = [
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

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET_COL], test_size=0.2, random_state=42
)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

param_grid = {
    'C': [0.01, 0.1, 0.5, 1.0, 10.0]
}


print("Finding optimal hyperparameters...")
grid_search = GridSearchCV(
    PassiveAggressiveRegressor(max_iter=1000, tol=1e-3, shuffle=True),
    param_grid, cv=5, scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)

# Get best model and parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

print("Training final model...")
model = PassiveAggressiveRegressor(
    C=best_params['C'], 
    max_iter=1000,
    tol=1e-3,
    shuffle=True
)

model.fit(X_train, y_train)

# Evaluate model
print("Evaluating model...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"RMSE: {rmse:.4f}°C")
print(f"MAE: {mae:.4f}°C")
print(f"R²: {r2:.4f}")

# Save the model
dump(model, "paper_model.joblib")
print("Model saved as paper_model.joblib")

# Feature importance
coef_abs = np.abs(model.coef_)
feature_importance = dict(zip(FEATURES, coef_abs))
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

print("\nTop 5 most important features:")
for feature, importance in sorted_importance[:5]:
    print(f"{feature}: {importance:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual Temperature (°C)')
plt.ylabel('Predicted Temperature (°C)')
plt.title('Actual vs Predicted CPU Temperature')
plt.tight_layout()
plt.savefig('model_evaluation.png')
plt.close()
print("Evaluation plot saved as model_evaluation.png")