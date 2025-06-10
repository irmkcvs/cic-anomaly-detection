
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.dropna()
    df['Label'] = df['Label'].apply(lambda x: 0 if x != 'Benign' else 1)

    X = df[["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts"]]
    y = df["Label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
