# Script to implement Collaborative Filtering using MovieLens dataset with PySpark (Approach for bigger datasets)
# Datasets used: 1M and 10M 

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, sum as _sum, abs as sql_abs, pow, avg, first
from pyspark.ml.feature import MinHashLSH, CountVectorizer
from pyspark.sql.window import Window
import csv
from pyspark.sql.functions import broadcast
from pyspark.sql.functions import isnan, isnull
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

spark = SparkSession.builder \
    .appName("MovieLensCF_Optimized") \
    .config("spark.executor.memory", "10g") \
    .config("spark.driver.memory", "10g") \
    .config("spark.local.dir", "/mnt/c/Users/diogo/Spark/spark-temp") \
    .getOrCreate()

# 1. Load datasets (ratings and movies)
ratings = spark.read.csv("ml-10M/ratings.dat", sep="::", header=True, inferSchema=True)
moviesCSV = spark.read.csv("ml-10M/movies.dat", sep="::", header=True, inferSchema=True)

ratings = ratings.toDF("userId", "movieId", "rating", "timestamp")
moviesCSV = moviesCSV.toDF("movieId", "title", "genres")

ratings = ratings.withColumn("userId", col("userId").cast("int")) \
                 .withColumn("movieId", col("movieId").cast("int")) \
                 .withColumn("rating", col("rating").cast("float"))

moviesCSV = moviesCSV.withColumn("movieId", col("movieId").cast("int")) \
                     .withColumn("title", col("title").cast("string")) \
                     .withColumn("genres", col("genres").cast("string"))

# 2. Split ratings into training and validation sets (90% / 10%)
train, val = ratings.randomSplit([0.9, 0.1], seed=42)
train = train.cache()

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
    ).cache()

# 7. Sample 10% of users for recommendations and validation
sampled_users = train.select("userId").distinct().sample(fraction=0.1, seed=42)
train_sampled = train.join(sampled_users, on="userId", how="inner").cache()
val_sampled = val.join(sampled_users, on="userId", how="inner").cache()

# 8. Generate rating predictions based on similar movies
user_rated_movies = train_sampled.select("userId", "movieId")

# 8.1. Join each user's rated movie with its similar neighbors
candidates_with_sim = user_rated_movies.join(similar_movies, user_rated_movies.movieId == similar_movies.movie1) \
    .select(col("userId"), col("movie1").alias("targetMovie"), col("movie2").alias("neighborMovie"), col("similarity"))

# 8.2. Join to get the neighbor's rating by the user
candidates_with_ratings = candidates_with_sim.join(
    train_sampled.withColumnRenamed("movieId", "neighborMovie"),
    on=["userId", "neighborMovie"]
).select("userId", "targetMovie", "neighborMovie", "similarity", "rating")

# 8.3. Compute weighted rating for each target movie
weighted_scores = candidates_with_ratings.withColumn("weighted_rating", col("similarity") * col("rating")) \
    .withColumn("abs_similarity", sql_abs(col("similarity")))

# 8.4. Aggregate to get final prediction per movie-user
predictions = weighted_scores.groupBy("userId", "targetMovie").agg(
    (_sum("weighted_rating") / _sum("abs_similarity")).alias("predicted_rating")
)

# 9. Attach movie titles and select top 10 predictions per user
predictions_mt = predictions.join(broadcast(moviesCSV), predictions.targetMovie == moviesCSV.movieId, "left") \
    .select("userId", "targetMovie", "title", "predicted_rating")

# 9.1. Rank predictions per user and keep top 10
windowSpec = Window.partitionBy("userId").orderBy(col("predicted_rating").desc())
predictions_top10 = predictions_mt.withColumn("rank", row_number().over(windowSpec)) \
    .filter(col("rank") <= 10).drop("rank")

# 10. Save results to CSV
predictions_top10.orderBy("userId", "predicted_rating", ascending=[True, False]) \
    .coalesce(1) \
    .write.option("header", True).mode("overwrite").csv("output_top10")

output_rows = predictions_top10.orderBy("userId", "predicted_rating", ascending=[True, False]).collect()
with open("output1M.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["UserID", "MovieID", "Movie Name", "Predicted Rating"])
    for row in output_rows:
        writer.writerow([row["userId"], row["targetMovie"], row["title"], f"{row['predicted_rating']:.4f}"])

# 11. Evaluate predictions using the validation set
# 11.1 Match validation ratings with similar movies from training
val_candidates_with_sim = val_sampled.join(similar_movies, val_sampled.movieId == similar_movies.movie1) \
    .select("userId", col("movieId").alias("targetMovie"), col("movie2").alias("neighborMovie"), "similarity", col("rating").alias("true_rating"))

# 11.2 Find ratings for similar movies from training set
val_candidates_with_ratings = val_candidates_with_sim.join(
    train_sampled.withColumnRenamed("movieId", "neighborMovie"),
    on=["userId", "neighborMovie"]
).select("userId", "targetMovie", "neighborMovie", "similarity", "rating", "true_rating")

# 11.3 Calculate weighted prediction
val_weighted_scores = val_candidates_with_ratings.withColumn("weighted_rating", col("similarity") * col("rating")) \
    .withColumn("abs_similarity", sql_abs(col("similarity")))

# Preserve the original true rating for each (user, movie) pair
window_spec = Window.partitionBy("userId", "targetMovie")
val_with_true = val_weighted_scores.withColumn("true_rating_preserved", first("true_rating").over(window_spec))

# Aggregate to final predictions and pair with original ratings
val_predictions = val_with_true.groupBy("userId", "targetMovie").agg(
    (_sum("weighted_rating") / _sum("abs_similarity")).alias("predicted_rating"),
    first("true_rating_preserved").alias("true_rating")
)

# Compute absolute and squared errors
val_with_errors = val_predictions.withColumn("abs_error", sql_abs(col("true_rating") - col("predicted_rating"))) \
    .withColumn("squared_error", pow(col("true_rating") - col("predicted_rating"), 2))

count_val_errors = val_with_errors.count()
print(f"Number of rows for evaluation: {count_val_errors}")

if count_val_errors == 0:
    print("Warning: No data to compute evaluation metrics. MAE and RMSE cannot be computed.")
    mae = float('nan')
    rmse = float('nan')
else:
    null_abs_error_count = val_with_errors.filter(isnull("abs_error") | isnan("abs_error")).count()
    null_squared_error_count = val_with_errors.filter(isnull("squared_error") | isnan("squared_error")).count()

    print(f"Rows with null/NaN abs_error: {null_abs_error_count}")
    print(f"Rows with null/NaN squared_error: {null_squared_error_count}")

    val_with_errors = val_with_errors.fillna({'abs_error': 0.0, 'squared_error': 0.0})

    # Compute MAE
    mae = val_with_errors.agg(avg("abs_error").alias("MAE")).collect()[0]["MAE"]

    # Compute RMSE from MSE
    mse = val_with_errors.agg(avg("squared_error").alias("MSE")).collect()[0]["MSE"]
    if mse is not None:
        rmse = mse ** 0.5
    else:
        print("MSE returned None — unable to compute RMSE.")
        rmse = float('nan')

print(f"\n=== Evaluation Metrics ===")
print(f"MAE  (Mean Absolute Error): {mae if mae == mae else 'NaN'}")  
print(f"RMSE (Root Mean Squared Error): {rmse if rmse == rmse else 'NaN'}")