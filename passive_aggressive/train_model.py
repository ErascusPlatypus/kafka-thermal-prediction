import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from joblib import dump

df = pd.read_csv("../../datasets/paper-datasets/train_data.csv", index_col=0)

FEATURES = [
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

model = PassiveAggressiveRegressor(
    C=0.1,  
    max_iter=1000,
    tol=1e-3,
    shuffle=True
)
model.fit(df[FEATURES], df[TARGET_COL])

dump(model, "paper_model.joblib")