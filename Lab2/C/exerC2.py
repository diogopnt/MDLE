# BFR algorithm using PySpark for large-scale clustering
# This version loads the FMA features dataset and performs clustering in chunks
# Adapt this if you're working in distributed environments or clusters

from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import KMeans
import numpy as np
import pandas as pd

sc = SparkContext(appName="BFR-Clustering")

DATA_PATH = "fma_metadata/features.csv"
K = 10  # best value from C1, adjust if needed
CHUNK_SIZE = 1000  # number of points per chunk (simulate incremental batches)

raw_lines = sc.textFile(DATA_PATH)
header_lines = raw_lines.zipWithIndex().filter(lambda x: x[1] < 3).map(lambda x: x[0])
cleaned_lines = raw_lines.zipWithIndex().filter(lambda x: x[1] >= 3).map(lambda x: x[0])

# Remove rows with non-numeric content
def parse_line(line):
    try:
        parts = [float(x) for x in line.split(",")]
        return parts
    except:
        return None

parsed_data = cleaned_lines.map(parse_line).filter(lambda x: x is not None)
data = parsed_data.zipWithIndex().map(lambda x: (x[1], x[0]))  # (index, features)
total_points = data.count()

# ==================== BFR STRUCTURES ====================
DS = {}  # Discard Set: {cluster_id: (N, SUM, SUMSQ)}
CS = {}  # Compression Set: {cluster_id: (N, SUM, SUMSQ)}
RS = []  # Retained Set: list of vectors (outliers)
point_assignment = {}  # {point_id: cluster_id}

# ==================== DISTANCE AND STATS UTILS ====================
def mahalanobis_distance(point, stats):
    N, SUM, SUMSQ = stats
    centroid = np.array(SUM) / N
    variance = np.array(SUMSQ) / N - (centroid ** 2)
    variance[variance == 0] = 1e-10
    return np.sqrt(np.sum(((point - centroid) ** 2) / variance))

def add_to_cluster(cluster_dict, cluster_id, point):
    N, SUM, SUMSQ = cluster_dict.get(cluster_id, (0, np.zeros(len(point)), np.zeros(len(point))))
    N += 1
    SUM += point
    SUMSQ += point ** 2
    cluster_dict[cluster_id] = (N, SUM, SUMSQ)

# ==================== PROCESS IN CHUNKS ====================
for i in range(0, total_points, CHUNK_SIZE):
    chunk = data.filter(lambda x: i <= x[0] < i + CHUNK_SIZE).map(lambda x: x[1]).collect()
    chunk_np = np.array(chunk)

    if len(DS) == 0:
        # Initial run — cluster the first chunk
        vectors = [Vectors.dense(x) for x in chunk_np]
        model = KMeans.train(sc.parallelize(vectors), K, maxIterations=30)

        # Build DS from KMeans output
        labels = [model.predict(v) for v in vectors]
        for label, point in zip(labels, chunk_np):
            add_to_cluster(DS, label, point)
    else:
        # For each point in chunk, assign to nearest DS cluster (if close enough)
        for point in chunk_np:
            assigned = False
            for cluster_id, stats in DS.items():
                if mahalanobis_distance(point, stats) < 2:  # threshold can be tuned
                    add_to_cluster(DS, cluster_id, point)
                    assigned = True
                    break

            if not assigned:
                # Try to add to CS
                for cluster_id, stats in CS.items():
                    if mahalanobis_distance(point, stats) < 2:
                        add_to_cluster(CS, cluster_id, point)
                        assigned = True
                        break

            if not assigned:
                RS.append(point)

        # If RS is large, recluster it
        if len(RS) > 2 * K:
            rs_vectors = [Vectors.dense(x) for x in RS]
            rs_model = KMeans.train(sc.parallelize(rs_vectors), K, maxIterations=10)
            labels = [rs_model.predict(v) for v in rs_vectors]

            RS = []  # reset
            new_CS = {}
            for label, point in zip(labels, rs_vectors):
                add_to_cluster(new_CS, label, np.array(point))
            CS.update(new_CS)

# ==================== FINAL MERGE ====================
# Attempt to merge CS clusters into DS
for cs_id, stats in list(CS.items()):
    for ds_id, ds_stats in DS.items():
        if mahalanobis_distance(np.array(stats[1]) / stats[0], ds_stats) < 2:
            add_to_cluster(DS, ds_id, np.array(stats[1]) / stats[0])
            del CS[cs_id]
            break

print("\n=== FINAL REPORT ===")
print(f"DS clusters: {len(DS)}")
print(f"CS clusters: {len(CS)}")
print(f"RS points (outliers): {len(RS)}")

sc.stop()