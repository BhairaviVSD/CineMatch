#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.mllib.evaluation import RankingMetrics


def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    # Loading in the train, val, and test splits
    train_split = spark.read.parquet(f'hdfs:/user/{userID}/train_split_latest.parquet', header=True, inferSchema=True)
    val_split = spark.read.parquet(f'hdfs:/user/{userID}/val_split_latest.parquet', header=True, inferSchema=True)
    test_split = spark.read.parquet(f'hdfs:/user/{userID}/test_split_latest.parquet', header=True, inferSchema=True)

    print('Printing dataset inferred schema')
    train_split.printSchema()

    # Extracting the utility scores and finding the top 100 movies    
    with_utility_scores = train_split.groupBy('movieId').agg(F.count('rating').alias('num_ratings'),
                F.sum('rating').alias('sum_ratings')).withColumn('utility_score', F.col('sum_ratings')/(F.col('num_ratings') + 1000))
    top_100_movies = with_utility_scores.orderBy(F.col('utility_score').desc()).limit(100)
    top_100_movies.show()

    # Extracting the top 100 recommendations from the dataframe and making a list
    top_100_list = top_100_movies.select('movieId').rdd.flatMap(lambda x: x).collect()
    top_100_col = F.array([F.lit(movie_id) for movie_id in top_100_list])

    # Creating predictions & ground-truth dataframe for metrics computation 
    val_ratings = val_split.groupBy("userId").agg(F.collect_list("movieId").alias("movieIds"))
    val_preds_and_labels = val_ratings.select(top_100_col.alias('predictions'), 'movieIds')
    val_preds_and_labels_rdd = val_preds_and_labels.rdd.map(lambda row: (row.predictions, row.movieIds))

    # Calculating the evaluation metrics
    val_metrics = RankingMetrics(val_preds_and_labels_rdd)
    print(f"Popularity Baseline, Validation Dataset: Precision at k=100 = {val_metrics.precisionAt(100)}")
    print(f"Popularity Baseline, Validation Dataset: Mean Average Precision (MAP) = {val_metrics.meanAveragePrecision}")
    print(f"Popularity Baseline, Validation Dataset: Mean Average Precision (MAP) at k=100 = {val_metrics.meanAveragePrecisionAt(100)}")
    print(f"Popularity Baseline, Validation Dataset: Recall ak k=100 = {val_metrics.recallAt(100)}")
    print(f"Popularity Baseline, Validation Dataset: NDCG at k=100 = {val_metrics.ndcgAt(100)}")


    # Now repeating the process carried out earlier, but on test dataset
    test_ratings = test_split.groupBy("userId").agg(F.collect_list("movieId").alias("movieIds"))
    test_preds_and_labels = test_ratings.select(top_100_col.alias('predictions'), 'movieIds')
    test_preds_and_labels_rdd = test_preds_and_labels.rdd.map(lambda row: (row.predictions, row.movieIds))

    test_metrics = RankingMetrics(test_preds_and_labels_rdd)
    print(f"Popularity Baseline, Test Dataset: Precision at k=100 = {test_metrics.precisionAt(100)}")
    print(f"Popularity Baseline, Test Dataset: Mean Average Precision (MAP) = {test_metrics.meanAveragePrecision}")
    print(f"Popularity Baseline, Test Dataset: Mean Average Precision (MAP) at k=100 = {test_metrics.meanAveragePrecisionAt(100)}")
    print(f"Popularity Baseline, Test Dataset: Recall ak k=100 = {test_metrics.recallAt(100)}")
    print(f"Popularity Baseline, Test Dataset: NDCG at k=100 = {test_metrics.ndcgAt(100)}")  


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('popularity_baseline').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)


