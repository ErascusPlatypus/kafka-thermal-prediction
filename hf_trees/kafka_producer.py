from kafka import KafkaProducer
import pandas as pd
import time
import json

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

test_df = pd.read_csv("../../datasets/paper-datasets/test_data.csv", index_col=0)
for _, row in test_df.iterrows():
    producer.send("sensor-data", value=row.to_dict())
    time.sleep(0.2)  