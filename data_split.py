#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
import pyspark.sql.functions as F


def data_split(spark, userID):
    '''Referred from Lab5 Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''

    ######### Dont Forget to change input dataset path #############
    ratings = spark.read.csv(f'hdfs:/user/{userID}/target/ratings.csv', header=True, inferSchema=True)
    ratings= ratings.na.drop()

    print('Filtering out irrelevant data')

    movie_counts = ratings.groupby('movieId').count()
    movies_to_keep = movie_counts.filter(movie_counts['count'] >= 20).select('movieId')
    base_ratings = ratings.join(movies_to_keep, 'movieId', 'inner')

    user_counts = base_ratings.groupby('userId').count()
    users_to_keep = user_counts.filter(user_counts['count'] >= 20).select('userId')
    base_ratings = base_ratings.join(users_to_keep, 'userId', 'inner')

    print('Total number of records:', base_ratings.count())
    print('Starting the Data Paritition Process...')

    user_partitions = base_ratings.select('userId').distinct().randomSplit([0.5, 0.25, 0.25], seed=42)

    train_usrids = user_partitions[0].select('userId')
    val_usrids = user_partitions[1].select('userId')
    test_usrids = user_partitions[2].select('userId')
    train_usrids_broadcast = F.broadcast(train_usrids)
    val_usrids_broadcast = F.broadcast(val_usrids)
    test_usrids_broadcast = F.broadcast(test_usrids)

    train = base_ratings.join(train_usrids_broadcast, 'userId', 'inner')
    val = base_ratings.join(val_usrids_broadcast, 'userId', 'inner')
    test = base_ratings.join(test_usrids_broadcast, 'userId', 'inner')

    print('Gathering historical data of users in val and test...')
    test_threshold = test.groupBy("userId").agg(F.expr("percentile_approx(timestamp, 0.6)").alias("threshold"))
    val_threshold = val.groupBy("userId").agg(F.expr("percentile_approx(timestamp, 0.6)").alias("threshold"))

    test_train = test.join(test_threshold, "userId").filter("timestamp <= threshold").drop("threshold")
    val_train = val.join(val_threshold, "userId").filter("timestamp <= threshold").drop("threshold")

    print('Adding the gathered historical data to train and removing them from val and test...')
    test = test.exceptAll(test_train)
    val = val.exceptAll(val_train)

    train = train.unionAll(test_train.select("userId", "movieId", "rating", "timestamp")) \
             .unionAll(val_train.select("userId", "movieId", "rating", "timestamp"))

    print('Number of Records in train split:',train.count())
    print('Number of Records in val split:',val.count())
    print('Number of Records in test split:',test.count())

    print('Writing the Parquet Files')

    train.write.mode('overwrite').parquet(f'hdfs:/user/{userID}/train_split.parquet')
    val.write.mode('overwrite').parquet(f'hdfs:/user/{userID}/val_split.parquet')
    test.write.mode('overwrite').parquet(f'hdfs:/user/{userID}/test_split.parquet')
    print('The Parquet files have been stored on hadoop.')


# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('data_split').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    data_split(spark, userID)
