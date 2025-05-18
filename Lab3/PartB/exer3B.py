from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sqrt, sum as _sum, count

spark = SparkSession.builder.appName("MovieLensCF").getOrCreate()

# 1. Read the ratings data
ratings = spark.read.csv("ml-latest-small/ratings.csv", header=True, inferSchema=True)

ratings.show(5)

# 2. Divide the data into training and validation sets
train, val = ratings.randomSplit([0.9, 0.1], seed=42)

print(f"Train: {train.count()} lines, Validation: {val.count()} lines")

# 3.1 Duplicate the ratings DataFrame to create pares of films per user 
ratings1 = train.select(col("userId").alias("userId"), 
                          col("movieId").alias("movieId1"), 
                          col("rating").alias("rating1"))

ratings2 = train.select(col("userId").alias("userId"),
                      col("movieId").alias("movieId2"), 
                      col("rating").alias("rating2"))

# 3.2 Join every pair of movies rated by the same user
movie_pairs = ratings1.join(ratings2, on="userId") \
                      .filter(col("movie1") < col("movie2"))
                      
# 3.3 Calculate the components of the cosine similarity
pair_scores = movie_pairs.withColumn("rating1_sq", pow(col("rating1"), 2)) \
                         .withColumn("rating2_sq", pow(col("rating2"), 2)) \
                         .withColumn("product", col("rating1") * col("rating2")) \
                             
# 3.4 Group by movie pairs and calculate the sums
similarity_stats = pair_scores.groupBy("movieId1", "movieId2") \
    .agg(_sum("product").alias("dot_product"),
         _sum("rating1_sq").alias("rating1_norm"),
         _sum("rating2_sq").alias("rating2_norm"),
         count("*").alias("num_common_users"))
    
# 3.5 Calculate the cosine similarity    
similarities = similarity_stats.withColumn(
    "similarity",
    col("dot_product") / (sqrt(col("rating1_norm")) * sqrt(col("rating2_norm")))
)



