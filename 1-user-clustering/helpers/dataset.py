from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def load_datasets(folder: str, paths: list[str], extension: str):
    datasets = []
    for path in paths:
        location = folder + "/" + path + "." + extension
        ds = pd.read_parquet(location)
        datasets.append(ds)
        head = ds.head()
        print('Dataset: ', path)
        column_names = ds.columns.tolist()
        print(column_names)
        print(head)
    return datasets


def calculate_elbow_wcss(scaled_features, max_k=10):
    wcss = []
    for k in range(1, max_k):  # Try different k values
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)  # Inertia = sum of squared distances
    return wcss


def calculate_elbow(dataframe: pd.DataFrame, scaled_features: np.ndarray):
    # Apply K-Means clustering

    wcss = calculate_elbow_wcss(scaled_features)

    # Plot the Elbow curve
    plt.plot(range(1, 10), wcss, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.show()
    k = 4  # Based on elbow method
    kmeans = KMeans(n_clusters=k, random_state=42)
    dataframe["Cluster"] = kmeans.fit_predict(scaled_features)
    return dataframe
