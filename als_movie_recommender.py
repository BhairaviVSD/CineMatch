from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rank
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator, RankingEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import pyspark.sql.functions as func
from pyspark.sql import Window

def als_movie_recommendation(spark, train_path, val_path):
    # Read the training and validation datasets
    train_data = spark.read.parquet(train_path, inferSchema=True)
    val_data = spark.read.parquet(val_path, inferSchema=True)
    
    # Cast columns to appropriate data types
    train_data = train_data.withColumn('userId', col('userId').cast('integer')) \
                           .withColumn('movieId', col('movieId').cast('integer')) \
                           .withColumn('rating', col('rating').cast('float')) \
                           .drop('timestamp')
    val_data = val_data.withColumn('userId', col('userId').cast('integer')) \
                       .withColumn('movieId', col('movieId').cast('integer')) \
                       .withColumn('rating', col('rating').cast('float')) \
                       .drop('timestamp')

    # Define hyperparameters
    rank_values = [10, 50, 100]
    reg_values = [0.01, 0.1, 1]

    # Initialize ALS model
    als = ALS(maxIter=5, userCol="userId", itemCol="movieId", ratingCol="rating",
              nonnegative=True, implicitPrefs=False, coldStartStrategy="drop")
    
    # Define parameter grid for cross-validation
    param_grid = ParamGridBuilder().addGrid(als.rank, rank_values) \
                                    .addGrid(als.regParam, reg_values) \
                                    .build()

    # Define evaluators
    rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    ranking_evaluator = RankingEvaluator(predictionCol='pred_movies', labelCol='movies', metricName="meanAveragePrecision")

    # Initialize cross-validator
    cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=rmse_evaluator, numFolds=5)

    # Train ALS model
    model = cv.fit(train_data)

    # Get the best model
    best_model = model.bestModel
    
    # Print best model parameters
    print("Best Model - Rank:", best_model.rank, " RegParam:", best_model._java_obj.parent().getRegParam())
    
    # Generate predictions on validation data
    val_predictions = model.transform(val_data)
    
    # Evaluate RMSE
    rmse = rmse_evaluator.evaluate(val_predictions)
    print("RMSE:", rmse)
    
    # Evaluate ranking metrics
    window = Window.partitionBy(val_predictions['userId']).orderBy(val_predictions['prediction'].desc())  
    val_predictions = val_predictions.withColumn('rank', rank().over(window)) \
                                     .filter(col('rank') <= 100) \
                                     .groupby("userId") \
                                     .agg(func.collect_list(val_predictions['movieId'].cast('double')).alias('pred_movies'))

    window = Window.partitionBy(val_data['userId']).orderBy(val_data['rating'].desc())  
    df_movies = val_data.withColumn('rank', rank().over(window)) \
                        .filter(col('rank') <= 100) \
                        .groupby("userId") \
                        .agg(func.collect_list(val_data['movieId'].cast('double')).alias('movies'))
    
    val_predictions = val_predictions.join(df_movies, val_predictions.userId==df_movies.userId).drop('userId')
    
    metrics = ['meanAveragePrecision','meanAveragePrecisionAtK','precisionAtK','ndcgAtK','recallAtK']
    metricsDict = {
        'rmse':rmse
    }
    for metric in metrics:
        rEvaluator = RankingEvaluator(predictionCol='pred_movies', labelCol='movies', metricName=metric)
        metricsDict[metric] = rEvaluator.evaluate(val_predictions)
        
    print(metricsDict)


if __name__ == "__main__":
    spark_session = SparkSession.builder.appName('Personalized Movie Recommendation').getOrCreate()
    train_path = "hdfs:/user/bvs9764_nyu_edu/train_split_small.parquet"
    val_path =  "hdfs:/user/bvs9764_nyu_edu/val_split_small.parquet"
    als_movie_recommendation(spark_session, train_path, val_path)