from pyspark import SparkContext
import json
import pandas as pd

sc = SparkContext(appName="C3-Genre-Analysis")

# Load cluster assignments (from C2) 
with open("track_cluster_map.json", "r") as f:
    cluster_map = {int(k): v for k, v in json.load(f).items()}  # track_id → cluster_id

# Load genre information from tracks.csv
tracks = pd.read_csv("fma_metadata/tracks.csv", header=[0, 1], index_col=0, low_memory=False)
track_genres = tracks[('track', 'genre_top')]  # Series: index = track_id, value = genre (or NaN)

# Convert data to PySpark RDDs
# Full genre RDD (including NaNs)
genre_rdd_full = sc.parallelize(track_genres.items())  # (track_id, genre or NaN)

# Cluster RDD from cluster_map
cluster_rdd = sc.parallelize(cluster_map.items())  # (track_id, cluster_id)

# Join genre and cluster info 
joined_rdd = cluster_rdd.join(genre_rdd_full)  # (track_id, (cluster_id, genre))

# Separate those with genre and those without
valid_genre_rdd = joined_rdd.filter(lambda x: pd.notna(x[1][1])) \
                            .map(lambda x: (x[1][0], x[1][1]))  
missing_genre_rdd = joined_rdd.filter(lambda x: pd.isna(x[1][1])) \
                              .map(lambda x: (x[1][0], 1)) 

# Count genres per cluster 
# Step 1: Count genre frequencies per cluster
genre_counts = (
    valid_genre_rdd
    .map(lambda x: ((x[0], x[1]), 1))  
    .reduceByKey(lambda a, b: a + b)
    .map(lambda x: (x[0][0], (x[0][1], x[1])))  
    .groupByKey()
    .mapValues(lambda items: sorted(items, key=lambda x: -x[1])[:5])  # Top 5 genres
)

# Step 2: Count (no genre) songs per cluster
missing_counts = missing_genre_rdd.reduceByKey(lambda a, b: a + b)  # (cluster_id, count)

genre_dict = dict(genre_counts.collect())        # cluster_id → list of (genre, count)
missing_dict = dict(missing_counts.collect())    # cluster_id → count

with open("C3_output.txt", "w") as f_out:
    cluster_ids = set(genre_dict.keys()).union(missing_dict.keys())
    ds_ids = sorted([cid for cid in cluster_ids if isinstance(cid, int)])
    cs_ids = sorted([cid for cid in cluster_ids if isinstance(cid, str)])

    for cluster_id in ds_ids + cs_ids:

        f_out.write(f"Cluster {cluster_id}:\n")

        if cluster_id in genre_dict:
            for genre, count in genre_dict[cluster_id]:
                f_out.write(f"  {genre}: {count} songs\n")

        if cluster_id in missing_dict:
            f_out.write(f"  ( with no genre): {missing_dict[cluster_id]} songs\n")

        f_out.write("\n")

# Debug/Info
print(f"Map entries: {list(cluster_map.items())[:5]}")
print(f"IDs map: {len(set(cluster_map.keys()))}")
print(f"IDs CSV: {len(track_genres.index)}")
print(f"Total available genres: {track_genres.notna().sum()}")

sc.stop()
