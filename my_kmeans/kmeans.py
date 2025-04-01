import numpy as np
import matplotlib.pyplot as plt

def distance(point1, point2):
    return np.sum((point1 - point2) ** 2)  # Squared Euclidean distance

def assign_clusters(data, means):
    clusters = [[] for _ in range(means.shape[1])]
    assignments = []
    for i in range(data.shape[1]):
        point = data[:, i]
        distances = [distance(point, means[:, j]) for j in range(means.shape[1])]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)
        assignments.append(cluster_index)
    return clusters, np.array(assignments)

def update_means(clusters, previous_means):
    new_means = previous_means.copy()
    for i, cluster in enumerate(clusters):
        if cluster:
            new_means[:, i] = np.mean(np.vstack(cluster).T, axis=1)
    return new_means

def calculate_wcss(data, means, assignments):
    wcss = 0
    for i in range(data.shape[1]):
        point = data[:, i]
        wcss += distance(point, means[:, assignments[i]])
    return wcss

import numpy as np
import pandas as pd

def k_means(data, k, max_iterations=100, random_state=42):
    np.random.seed(random_state)  # Set seed for reproducibility

    # Convert DataFrame to NumPy array if needed
    if isinstance(data, pd.DataFrame):
        data = data.values  

    initial_indices = np.random.choice(data.shape[1], k, replace=False)
    means = data[:, initial_indices].copy()

    if max_iterations is None:
        while True:  # Infinite loop with a valid termination condition
            clusters, assignments = assign_clusters(data, means)
            new_means = update_means(clusters, means)
            if np.allclose(means, new_means):
                break
            means = new_means
    else:
        for _ in range(max_iterations):
            clusters, assignments = assign_clusters(data, means)
            new_means = update_means(clusters, means)
            if np.allclose(means, new_means):
                break
            means = new_means

    wcss = calculate_wcss(data, means, assignments)  # Ensure this function is defined
    return means, wcss, assignments

