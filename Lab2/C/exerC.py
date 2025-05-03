import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

# Load features.csv with a multi-index header (3 levels: feature group, statistic, feature name)
features = pd.read_csv("fma_metadata/features.csv", index_col=0, header=[0, 1, 2])

# Load tracks.csv with a 2-level header (used to select subsets like "small", "medium", "large")
tracks = pd.read_csv("fma_metadata/tracks.csv", index_col=0, header=[0, 1])

# Filter to keep only the tracks in the 'small' subset
subset_mask = tracks[("set", "subset")] == "small"
features = features[subset_mask]

# Keep only columns where the second level of the header is 'mean'
# This ensures we use only the average values of the audio features
cols_mean = [col for col in features.columns if col[1] == "mean"]
features = features[cols_mean]

# Drop rows with missing values (NaNs) to avoid errors during clustering
features = features.dropna()

# Convert the cleaned DataFrame into a NumPy array for use in clustering
X = features.to_numpy()

def compute_cluster_metrics(X, labels, k):
    """
    Computes statistics per cluster:
    - number of points
    - radius (mean distance to centroid)
    - diameter (mean pairwise distance)
    - density based on radius and diameter
    """
    cluster_results = {}

    for label in np.unique(labels):
        cluster_points = X[labels == label]
        n = len(cluster_points)

        if n == 0:
            continue

        # Compute the centroid of the cluster
        centroid = np.mean(cluster_points, axis=0)

        # Radius: square root of mean squared distance to the centroid
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        r2 = np.mean(distances ** 2)
        radius = np.sqrt(r2)

        # Diameter: square root of mean squared pairwise distance within the cluster
        if n == 1:
            d2 = 0.0
            diameter = 0.0
        else:
            pairwise = pdist(cluster_points, metric="euclidean")
            d2 = np.mean(pairwise ** 2)
            diameter = np.sqrt(d2)

        # Densities: number of points per unit squared distance
        density_r = n / r2 if r2 > 0 else 0.0
        density_d = n / d2 if d2 > 0 else 0.0

        cluster_results[label] = {
            "n_points": n,
            "radius": radius,
            "diameter": diameter,
            "density_r": density_r,
            "density_d": density_d,
        }

    return cluster_results

# Dictionary to store global results per k (number of clusters)
results = {}

# Output file to store cluster statistics
with open("C1_cluster_stats.txt", "w") as f_out:
    for k in range(8, 17):  # Vary number of clusters from 8 to 16
        f_out.write(f"\n=== Clustering with k = {k} ===\n")
        print(f"\n=== Clustering with k = {k} ===")

        # Apply Agglomerative Clustering with Ward linkage
        clustering = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = clustering.fit_predict(X)

        # Compute metrics for each cluster
        cluster_stats = compute_cluster_metrics(X, labels, k)

        # Compute average metrics across all clusters for this k
        avg_radius = np.mean([v["radius"] for v in cluster_stats.values()])
        avg_diameter = np.mean([v["diameter"] for v in cluster_stats.values()])
        avg_density_r = np.mean([v["density_r"] for v in cluster_stats.values()])
        avg_density_d = np.mean([v["density_d"] for v in cluster_stats.values()])

        # Store results in dictionary for later analysis
        results[k] = {
            "radius": avg_radius,
            "diameter": avg_diameter,
            "density_r": avg_density_r,
            "density_d": avg_density_d,
            "cluster": cluster_stats,
        }

        # Write per-cluster statistics to output file and print them
        for cid, stat in cluster_stats.items():
            line = (
                f"Cluster {cid}: size={stat['n_points']}, "
                f"radius={stat['radius']:.2f}, diameter={stat['diameter']:.2f}, "
                f"density_r={stat['density_r']:.2f}, density_d={stat['density_d']:.2f}"
            )
            f_out.write(line + "\n")
            print(line)