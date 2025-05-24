from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_set, sum as _sum, abs as sql_abs, pow, avg, first
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.window import Window
import csv

#spark = SparkSession.builder.appName("MovieLensCF").getOrCreate()

spark = SparkSession.builder \
    .appName("MovieLensCF") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

#.config("spark.memory.fraction", "0.8") \

# 1. Read the ratings data
ratings = spark.read.csv("ml-1m/ratings.dat", sep="::",header=True, inferSchema=True)
moviesCSV = spark.read.csv("ml-1m/movies.dat", sep="::", header=True, inferSchema=True)

ratings = ratings.toDF("userId", "movieId", "rating", "timestamp")
moviesCSV = moviesCSV.toDF("movieId", "title", "genres")

ratings = ratings.withColumn("userId", col("userId").cast("int")) \
                 .withColumn("movieId", col("movieId").cast("int")) \
                 .withColumn("rating", col("rating").cast("float"))

moviesCSV = moviesCSV.withColumn("movieId", col("movieId").cast("int")) \
                     .withColumn("title", col("title").cast("string")) \
                     .withColumn("genres", col("genres").cast("string"))

ratings.show(5)
moviesCSV.show(5)

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

predictions_mt = predictions.join(moviesCSV, predictions.targetMovie == moviesCSV.movieId, "left") \
    .select(
        "userId",
        "targetMovie",
        "title",
        "predicted_rating"
    )
    
predictions_mt.orderBy("userId", "predicted_rating", ascending=False).show(50, truncate=False)

# 8. Export predictions to a .txt file
#output_rows = predictions.orderBy("userId", "targetMovie").collect()
output_rows = predictions_mt.orderBy("userId", "predicted_rating", ascending=[True, False]).collect()

with open("output3B2.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["UserID", "MovieID", "Movie Name", "Predicted Rating"])
    for row in output_rows:
        writer.writerow([row["userId"], row["targetMovie"], row["title"], f"{row['predicted_rating']:.4f}"])
        
# 9. Generate predictions for the pares of the validation set (user, movie)
val_candidates_with_sim = val.join(similar_movies, val.movieId == similar_movies.movie1) \
    .select(
        col("userId"),
        col("movieId").alias("targetMovie"),
        col("movie2").alias("neighborMovie"),
        col("similarity"),
        col("rating").alias("true_rating")
    )   
    
# 10. Get the ratings that user has given to the neighbor movie
val_candidates_with_ratings = val_candidates_with_sim.join(
    train.withColumnRenamed("movieId", "neighborMovie"),
    on=["userId", "neighborMovie"]
).select(
    col("userId"),
    col("targetMovie"),
    col("neighborMovie"),
    col("similarity"),
    col("rating"), # neighbor movie rating
    col("true_rating") # true rating for the target movie
)  

# 11. Calculate the weighted predicted rating  
val_weighted_scores = val_candidates_with_ratings.withColumn(
    "weighted_rating", col("similarity") * col("rating")
).withColumn(
    "abs_similarity", sql_abs(col("similarity"))
)

window_spec = Window.partitionBy("userId", "targetMovie")
val_with_true = val_weighted_scores.withColumn("true_rating_preserved", first("true_rating").over(window_spec))

val_predictions = val_with_true.groupBy("userId", "targetMovie").agg(
    (_sum("weighted_rating") / _sum("abs_similarity")).alias("predicted_rating"),
    first("true_rating_preserved").alias("true_rating")
)

# 12. Join with the true ratings to calculate RMSE and MAE
val_with_errors = val_predictions.withColumn(
    "abs_error", sql_abs(col("true_rating") - col("predicted_rating"))
).withColumn(
    "squared_error", pow(col("true_rating") - col("predicted_rating"), 2)
)

mae = val_with_errors.agg(avg("abs_error").alias("MAE")).collect()[0]["MAE"]
rmse = val_with_errors.agg(avg("squared_error").alias("RMSE")).collect()[0][0] ** 0.5

print(f"\n=== Evaluation Metrics ===")
print(f"MAE  (Mean Absolute Error): {mae:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")

# todo: A predicted rating cannot be more than 5
# todo: Only recommend N movies per user with the highest predicted ratings (Also an improvement - because of the .crossJoin)
# todo: Use parameters to choose N movies to recommend, number of hash tables, dataset to use and the threshold of similarity

## Improvements:
# Delete .show() calls to avoid printing large datasets
# Delete .count() calls to avoid counting large datasets
# Utilize caching for large datasets to speed up processing
# Try to read the data from the dat files in a more efficient way
# Maybe delete .collect() calls to avoid collecting large datasets to the driver