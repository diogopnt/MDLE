from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, sum as _sum
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import abs as sql_abs

#spark = SparkSession.builder.appName("MovieLensCF").getOrCreate()

spark = SparkSession.builder \
    .appName("MovieLensCF") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()


# 1. Read the ratings data
ratings = spark.read.csv("ml-latest-small/ratings.csv", header=True, inferSchema=True)
ratings.show(5)

# 2. Divide the data into training and validation sets
train, val = ratings.randomSplit([0.9, 0.1], seed=42)
print(f"Train: {train.count()} lines, Validation: {val.count()} lines")

# 3. Create a user-item matrix
movie_users = train.select("movieId", col("userId").cast("string").alias("userId")) \
    .groupBy("movieId") \
    .agg(collect_set("userId").alias("users"))
    
# 4. Convert the user lists into a binary vector
cv = CountVectorizer(inputCol="users", outputCol="features", binary=True)
cv_model = cv.fit(movie_users)
movie_vectors = cv_model.transform(movie_users)


# 5. Aplply MinHashLSH to find similar movies
mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
model = mh.fit(movie_vectors)

# 6. Find nearest neighbors 
similar_movies = model.approxSimilarityJoin(movie_vectors, movie_vectors, 1.0, distCol="JaccardDistance") \
    .filter(col("datasetA.movieId") < col("datasetB.movieId")) \
    .select(
        col("datasetA.movieId").alias("movie1"),
        col("datasetB.movieId").alias("movie2"),
        (1 - col("JaccardDistance")).alias("similarity")
    )
    
# 7. Calculate similarity scores (1 - JaccardDistance)
#similar_movies = similar_movies.withColumn("similarity", 1 - col("JaccardDistance"))

#similar_movies.orderBy(col("similarity").desc()).show(500, truncate=False)

# 7.1 Generate candidates (user, movie) pairs that don't exist
users = train.select("userId").distinct()
movies = train.select("movieId").distinct()
user_movie_candidates = users.crossJoin(movies)
existing_ratings = train.select("userId", "movieId")
candidates = user_movie_candidates.join(existing_ratings, ["userId", "movieId"], "left_anti")


# 7.2 Join with similar movies to get recommendations
candidates_with_sim = candidates.join(similar_movies, candidates.movieId == similar_movies.movie1) \
    .select(
        col("userId"),
        col("movie1").alias("targetMovie"),
        col("movie2").alias("neighborMovie"),
        col("similarity")
    )
    
# 7.3 Join with ratings that user has given to the neighbor movie
candidates_with_ratings = candidates_with_sim.join(
    train.withColumnRenamed("movieId", "neighborMovie"),
    on=["userId", "neighborMovie"]
).select(
    col("userId"),
    col("targetMovie"),
    col("neighborMovie"),
    col("similarity"),
    col("rating")
)

# 7.4 Calculate the weighted rating
weighted_scores = candidates_with_ratings.withColumn("weighted_rating", col("similarity") * col("rating")) \
    .withColumn("abs_similarity", sql_abs(col("similarity")))

predictions = weighted_scores.groupBy("userId", "targetMovie").agg(
    (_sum("weighted_rating") / _sum("abs_similarity")).alias("predicted_rating")
)

predictions.orderBy("userId", "targetMovie").show(50, truncate=False)

# 8. Export predictions to a .txt file
output_rows = predictions.orderBy("userId", "targetMovie").collect()

with open("predicted_ratings.txt", "w", encoding="utf-8") as f:
    for row in output_rows:
        line = f"{row['userId']}\t{row['targetMovie']}\t{row['predicted_rating']:.4f}\n"
        f.write(line)

# todo: Evaluate the model
# todo: Only recommend N movies per user with the highest predicted ratings
# todo: Use bigger datasets
# todo: Use parameters to choose N movies to recommend, number of hash tables, dataset to use and the threshold of similarity