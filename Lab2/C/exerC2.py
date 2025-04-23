from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.clustering import KMeans
import numpy as np

sc = SparkContext(appName="BFR-Clustering-Final")

# ============== CONFIG ==============
DATA_PATH = "fma_metadata/features.csv"
K = 10  # final number of clusters
CHUNK_SIZE = 1000  # number of data points per chunk
ALPHA = 2  # Mahalanobis threshold
INIT_MULTIPLIER = 5  # for initial KMeans (e.g., 5 * K)

log_lines = []  # To accumulate log lines for file output

# ============== HELPERS ==============
def mahalanobis_distance(point, stats):
    N, SUM, SUMSQ = stats
    centroid = SUM / N
    variance = (SUMSQ / N) - (centroid ** 2)
    variance[variance == 0] = 1e-10
    return np.sqrt(np.sum(((point - centroid) ** 2) / variance))

def add_to_cluster(cluster_dict, cluster_id, point):
    N, SUM, SUMSQ = cluster_dict.get(cluster_id, (0, np.zeros(len(point)), np.zeros(len(point))))
    N += 1
    SUM += point
    SUMSQ += point ** 2
    cluster_dict[cluster_id] = (N, SUM, SUMSQ)

def log_cluster_stats(name, clusters):
    log_lines.append(f"\n{name} contains {len(clusters)} clusters:")
    for cid, (N, SUM, SUMSQ) in clusters.items():
        log_lines.append(f"  Cluster {cid}: N={N}, SUM[:3]={SUM[:3]}, SUMSQ[:3]={SUMSQ[:3]}")

# ============== LOAD DATA ==============
raw_lines = sc.textFile(DATA_PATH)
cleaned_lines = raw_lines.zipWithIndex().filter(lambda x: x[1] >= 3).map(lambda x: x[0])

def parse_line(line):
    try:
        parts = [float(x) for x in line.split(",")]
        return parts
    except:
        return None

parsed_data = cleaned_lines.map(parse_line).filter(lambda x: x is not None)
data = parsed_data.zipWithIndex().map(lambda x: (x[1], x[0]))
total_points = data.count()

# ============== BFR STRUCTURES ==============
DS = {}  # Discard Set
CS = {}  # Compression Set
RS = []  # Retained Set

log_lines.append("\n=== Initialization ===")

# Get first chunk of data
initial_chunk = data.filter(lambda x: x[0] < CHUNK_SIZE).map(lambda x: x[1]).collect()
initial_np = np.array(initial_chunk)

# Run KMeans with many clusters to detect outliers
init_vectors = [Vectors.dense(x) for x in initial_np]
init_model = KMeans.train(sc.parallelize(init_vectors), K * INIT_MULTIPLIER, maxIterations=30)
init_labels = [init_model.predict(v) for v in init_vectors]

# Group points by cluster label
cluster_map = {}
for label, point in zip(init_labels, initial_np):
    cluster_map.setdefault(label, []).append(point)

# Separate small clusters to RS, others go to inliers
inliers = []
for label, points in cluster_map.items():
    if len(points) <= 2:
        RS.extend(points)
    else:
        inliers.extend(points)

# Cluster inliers into K clusters to form DS
inlier_vectors = [Vectors.dense(x) for x in inliers]
ds_model = KMeans.train(sc.parallelize(inlier_vectors), K, maxIterations=30)
ds_labels = [ds_model.predict(v) for v in inliers]
for label, point in zip(ds_labels, inliers):
    add_to_cluster(DS, label, point)

# Re-cluster RS into 5*K clusters to form CS (discard singletons)
rs_vectors = [Vectors.dense(x) for x in RS]
RS = []
if rs_vectors:
    rs_model = KMeans.train(sc.parallelize(rs_vectors), K * INIT_MULTIPLIER, maxIterations=10)
    rs_labels = [rs_model.predict(v) for v in rs_vectors]
    temp_clusters = {}
    for label, vec in zip(rs_labels, rs_vectors):
        temp_clusters.setdefault(label, []).append(np.array(vec))
    for label, points in temp_clusters.items():
        if len(points) > 1:
            for p in points:
                add_to_cluster(CS, f"CS_{label}", p)
        else:
            RS.extend(points)

# ============== CHUNK PROCESSING ==============
for i in range(CHUNK_SIZE, total_points, CHUNK_SIZE):
    chunk = data.filter(lambda x: i <= x[0] < i + CHUNK_SIZE).map(lambda x: x[1]).collect()
    chunk_np = np.array(chunk)
    log_lines.append(f"\n--- Processing chunk {i // CHUNK_SIZE + 1} ---")
    log_lines.append(f"Points in this chunk: {len(chunk_np)}")

    for point in chunk_np:
        assigned = False
        
        # Try assigning point to DS
        for cid, stats in DS.items():
            if mahalanobis_distance(point, stats) < ALPHA * np.sqrt(len(point)):
                add_to_cluster(DS, cid, point)
                assigned = True
                break
            
        # Try assigning point to CS if not in DS
        if not assigned:
            for cid, stats in CS.items():
                if mahalanobis_distance(point, stats) < ALPHA * np.sqrt(len(point)):
                    add_to_cluster(CS, cid, point)
                    assigned = True
                    break
                
        # Point remains unassigned → add to RS            
        if not assigned:
            RS.append(point)

    # Recluster RS if it becomes too large
    if len(RS) > 5 * K:
        rs_vectors = [Vectors.dense(x) for x in RS]
        rs_model = KMeans.train(sc.parallelize(rs_vectors), K * INIT_MULTIPLIER, maxIterations=10)
        rs_labels = [rs_model.predict(v) for v in rs_vectors]
        RS = []
        temp_clusters = {}
        for label, vec in zip(rs_labels, rs_vectors):
            temp_clusters.setdefault(label, []).append(np.array(vec))
        for label, points in temp_clusters.items():
            if len(points) > 1:
                for p in points:
                    add_to_cluster(CS, f"CS_{label}_{i}", p)
            else:
                RS.extend(points)

    # Merge CS clusters if they are too close (based on Mahalanobis distance)
    cs_keys = list(CS.keys())
    for i in range(len(cs_keys)):
        for j in range(i + 1, len(cs_keys)):
            id1, id2 = cs_keys[i], cs_keys[j]
            if id1 in CS and id2 in CS:
                dist = mahalanobis_distance(CS[id1][1] / CS[id1][0], CS[id2])
                if dist < ALPHA * np.sqrt(len(point)):
                    add_to_cluster(CS, id1, CS[id2][1] / CS[id2][0])
                    del CS[id2]

    log_lines.append(f"DS clusters: {len(DS)} ({sum([v[0] for v in DS.values()])} points)")
    log_lines.append(f"CS clusters: {len(CS)} ({sum([v[0] for v in CS.values()])} points)")
    log_lines.append(f"RS points: {len(RS)}")

# ============== FINAL MERGE ==============
for cid, stats in list(CS.items()):
    for dsid, ds_stats in DS.items():
        if mahalanobis_distance(stats[1] / stats[0], ds_stats) < ALPHA * np.sqrt(len(point)):
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

sc.stop()