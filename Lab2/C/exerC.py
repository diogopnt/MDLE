import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

# Load the features file, skipping the first 3 metadata lines
features = pd.read_csv("fma_metadata/features.csv", skiprows=3, header=None)

# Convert all columns to numeric (float), coercing errors to NaN
features = features.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values (ensure clean data for clustering)
features = features.dropna()

# Keep only the first 106574 tracks (to match track metadata)
features = features.iloc[:106574, :]

# Load track metadata with multi-level column headers
tracks = pd.read_csv("fma_metadata/tracks.csv", header=[0, 1], index_col=0, low_memory=False)

# Set the features dataframe index to match the track IDs
features.index = tracks.index

# Select the medium subset of 8000 tracks (as specified in the exercise)
medium_subset_ids = tracks[tracks[('set', 'subset')] == 'medium'].index

# Filter features to only include the medium subset tracks
features_filtered = features.loc[medium_subset_ids]

# Convert the features to a NumPy array for clustering
X = features_filtered.to_numpy()

def compute_cluster_stats(X, labels, k):
    stats = []

    for cluster_id in range(k):
        points = X[labels == cluster_id]
        if len(points) == 0:
            continue

        # Compute the centroid of the cluster
        centroid = np.mean(points, axis=0)

        # Radius: the maximum distance from any point to the centroid
        dists_to_centroid = np.linalg.norm(points - centroid, axis=1)
        radius = np.max(dists_to_centroid)

        # Diameter: the maximum distance between any two points in the cluster
        pairwise_dists = pairwise_distances(points)
        diameter = np.max(pairwise_dists)

        # Average squared distance from centroid (for density)
        r2 = np.mean(dists_to_centroid ** 2)

        # Average squared pairwise distance within the cluster
        d2 = np.mean(pairwise_dists ** 2)

        stats.append({
            "cluster": cluster_id,
            "size": len(points),
            "radius": radius,
            "diameter": diameter,
            "density_r2": r2,
            "density_d2": d2
        })

    return stats

with open("C1_cluster_stats.txt", "w") as f_out:
    for k in range(8, 17):
        print(f"\n=== Clustering with k = {k} ===")
        f_out.write(f"\n=== Clustering with k = {k} ===\n")

        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clustering.fit_predict(X)

        cluster_stats = compute_cluster_stats(X, labels, k)

        for stat in cluster_stats:
            line = (f"Cluster {stat['cluster']}: size={stat['size']}, "
                    f"radius={stat['radius']:.2f}, diameter={stat['diameter']:.2f}, "
                    f"density_r2={stat['density_r2']:.4f}, density_d2={stat['density_d2']:.4f}")
            
            print(line)
            f_out.write(line + "\n")
