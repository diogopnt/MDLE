from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import KMeans
import numpy as np
import json

sc = SparkContext(appName="BFR-Clustering-Final")

DATA_PATH = "fma_metadata/features.csv"
K = 10  # Number of final clusters
CHUNK_SIZE = 1000  # Number of points per chunk
ALPHA = 2  # Mahalanobis distance threshold multiplier
INIT_MULTIPLIER = 5  # Number of clusters in initial overclustering
log_lines = []  # Log messages will be stored here

# Compute Mahalanobis distance between a point and a cluster
def mahalanobis_distance(point, stats):
    N, SUM, SUMSQ = stats
    centroid = SUM / N
    variance = (SUMSQ / N) - (centroid ** 2)
    variance[variance == 0] = 1e-10  # Avoid division by zero
    return np.sqrt(np.sum(((point - centroid) ** 2) / variance))

# Update statistics (N, SUM, SUMSQ) for a cluster with a new point
def add_to_cluster(cluster_dict, cluster_id, point):
    N, SUM, SUMSQ = cluster_dict.get(cluster_id, (0, np.zeros(len(point)), np.zeros(len(point))))
    N += 1
    SUM += point
    SUMSQ += point ** 2
    cluster_dict[cluster_id] = (N, SUM, SUMSQ)

# Log cluster statistics for debugging and reporting
def log_cluster_stats(name, clusters):
    log_lines.append(f"\n{name} contains {len(clusters)} clusters:")
    for cid, (N, SUM, SUMSQ) in clusters.items():
        log_lines.append(f"  Cluster {cid}: N={N}, SUM[:3]={SUM[:3]}, SUMSQ[:3]={SUMSQ[:3]}")

# Read the data file
raw_lines = sc.textFile(DATA_PATH)

# Skip the header and metadata lines
cleaned_lines = raw_lines.zipWithIndex().filter(lambda x: x[1] >= 3).map(lambda x: x[0])

# Convert CSV lines to float lists
def parse_line(line):
    try:
        return [float(x) for x in line.split(",")]
    except:
        return None

parsed_data = cleaned_lines.map(parse_line).filter(lambda x: x is not None)

# Create RDD of (track_id, features)
data = parsed_data.zipWithIndex().map(lambda x: (x[1], x[0]))
total_points = data.count()

# BFR STRUCTURES 

DS = {}  # Discard Set: summarized clusters with many points
CS = {}  # Compression Set: summarized clusters with few points
RS = []  # Retained Set: unassigned outlier points
track_cluster_map = {}  # Mapping from track_id to final cluster

log_lines.append("\n=== Initialization ===")

# Get the initial chunk of data
initial_chunk = data.filter(lambda x: x[0] < CHUNK_SIZE).collect()
initial_np = np.array([x[1] for x in initial_chunk])
initial_ids = [x[0] for x in initial_chunk]

# Overcluster the initial chunk using KMeans with K * INIT_MULTIPLIER
init_vectors = [Vectors.dense(x) for x in initial_np]
init_model = KMeans.train(sc.parallelize(init_vectors), K * INIT_MULTIPLIER, maxIterations=30)
init_labels = [init_model.predict(v) for v in init_vectors]

# Group initial points by cluster label
cluster_map = {}
for idx, label in enumerate(init_labels):
    cluster_map.setdefault(label, []).append((initial_ids[idx], initial_np[idx]))

# Separate small clusters into RS, rest go to inliers
inliers = []
for label, points in cluster_map.items():
    if len(points) <= 2:
        RS.extend(points)
    else:
        inliers.extend(points)

# Cluster inliers into K clusters → form the DS
inlier_vectors = [Vectors.dense(x[1]) for x in inliers]
inlier_ids = [x[0] for x in inliers]
ds_model = KMeans.train(sc.parallelize(inlier_vectors), K, maxIterations=30)
ds_labels = [ds_model.predict(v) for v in inlier_vectors]

# Build DS from clustered inliers
for idx, point in enumerate(inliers):
    cluster_id = ds_labels[idx]
    point_id, vec = point
    add_to_cluster(DS, cluster_id, vec)
    track_cluster_map[int(point_id)] = int(cluster_id)

# Cluster the RS to form the CS (if possible)
RS = [(i, v) for i, v in RS]
rs_vectors = [Vectors.dense(x[1]) for x in RS]
rs_ids = [x[0] for x in RS]
RS = []  # Clear RS for re-use

if rs_vectors:
    rs_model = KMeans.train(sc.parallelize(rs_vectors), K * INIT_MULTIPLIER, maxIterations=10)
    rs_labels = [rs_model.predict(v) for v in rs_vectors]

    # Group RS points into temporary clusters
    temp_clusters = {}
    for idx, label in enumerate(rs_labels):
        temp_clusters.setdefault(label, []).append((rs_ids[idx], np.array(rs_vectors[idx])))

    # Points in small clusters remain in RS, others form CS
    for label, points in temp_clusters.items():
        if len(points) > 1:
            for pid, vec in points:
                cluster_name = f"CS_{label}"
                add_to_cluster(CS, cluster_name, vec)
                track_cluster_map[int(pid)] = cluster_name
        else:
            RS.extend(points)

for i in range(CHUNK_SIZE, total_points, CHUNK_SIZE):
    # Get the next data chunk
    chunk = data.filter(lambda x: i <= x[0] < i + CHUNK_SIZE).collect()
    chunk_ids = [x[0] for x in chunk]
    chunk_np = np.array([x[1] for x in chunk])

    log_lines.append(f"\n--- Processing chunk {i // CHUNK_SIZE + 1} ---")
    log_lines.append(f"Points in this chunk: {len(chunk_np)}")

    for pid, point in zip(chunk_ids, chunk_np):
        assigned = False

        # Try assigning point to DS
        for cid, stats in DS.items():
            if mahalanobis_distance(point, stats) < ALPHA * np.sqrt(len(point)):
                add_to_cluster(DS, cid, point)
                track_cluster_map[int(pid)] = int(cid)
                assigned = True
                break

        # If not assigned, try CS
        if not assigned:
            for cid, stats in CS.items():
                if mahalanobis_distance(point, stats) < ALPHA * np.sqrt(len(point)):
                    add_to_cluster(CS, cid, point)
                    track_cluster_map[int(pid)] = cid
                    assigned = True
                    break

        # If still not assigned, keep it in RS
        if not assigned:
            RS.append((pid, point))

    # Recluster RS if it becomes too large
    if len(RS) > 5 * K:
        rs_vectors = [Vectors.dense(x[1]) for x in RS]
        rs_ids = [x[0] for x in RS]
        RS = []
        rs_model = KMeans.train(sc.parallelize(rs_vectors), K * INIT_MULTIPLIER, maxIterations=10)
        rs_labels = [rs_model.predict(v) for v in rs_vectors]

        temp_clusters = {}
        for idx, label in enumerate(rs_labels):
            temp_clusters.setdefault(label, []).append((rs_ids[idx], np.array(rs_vectors[idx])))

        # Points forming valid clusters become part of CS
        for label, points in temp_clusters.items():
            if len(points) > 1:
                for pid, vec in points:
                    cluster_name = f"CS_{label}_{i}"
                    add_to_cluster(CS, cluster_name, vec)
                    track_cluster_map[int(pid)] = cluster_name
            else:
                RS.extend(points)

    log_lines.append(f"DS clusters: {len(DS)} ({sum([v[0] for v in DS.values()])} points)")
    log_lines.append(f"CS clusters: {len(CS)} ({sum([v[0] for v in CS.values()])} points)")
    log_lines.append(f"RS points: {len(RS)}")

# Final Merge
# Try merging CS clusters into DS if they are close enough
for cid, stats in list(CS.items()):
    for dsid, ds_stats in DS.items():
        if mahalanobis_distance(stats[1] / stats[0], ds_stats) < ALPHA * np.sqrt(len(stats[1])):
            add_to_cluster(DS, dsid, stats[1] / stats[0])
            del CS[cid]
            break

log_lines.append("\n=== FINAL REPORT ===")
log_lines.append(f"DS clusters: {len(DS)}")
log_lines.append(f"CS clusters: {len(CS)}")
log_lines.append(f"RS points: {len(RS)}")

log_cluster_stats("Discard Set (DS)", DS)
log_cluster_stats("Compression Set (CS)", CS)

with open("C2_output.txt", "w") as f:
    for line in log_lines:
        f.write(line + "\n")

with open("track_cluster_map.json", "w") as f_map:
    json.dump(track_cluster_map, f_map)

sc.stop()