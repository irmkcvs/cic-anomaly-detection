import pandas as pd
import gc

columns_to_use = ["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "Label"]

def load_cic_ids_data():
    chunk_size = 500000
    dataset_list = []

    for file in ["data/03-02-2018.csv", "data/03-01-2018.csv", "data/02-23-2018.csv", "data/02-14-2018.csv"]:
        for chunk in pd.read_csv(file, usecols=columns_to_use, chunksize=chunk_size, low_memory=False):
            chunk = chunk[chunk["Flow Duration"] != "Flow Duration"]
            dataset_list.append(chunk)

    dataset = pd.concat(dataset_list, ignore_index=True)

    dataset["Flow Duration"] = pd.to_numeric(dataset["Flow Duration"], errors="coerce")
    dataset["Tot Fwd Pkts"] = pd.to_numeric(dataset["Tot Fwd Pkts"], errors="coerce")
    dataset["Tot Bwd Pkts"] = pd.to_numeric(dataset["Tot Bwd Pkts"], errors="coerce")

    dataset = dataset.dropna()

    print("[DEBUG] Dataset satır sayısı:", dataset.shape)
    print("[DEBUG] Dataset sütunları:", dataset.columns.tolist())

    gc.collect()
    return dataset
