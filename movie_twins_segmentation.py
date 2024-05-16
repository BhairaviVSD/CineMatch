from pyspark.sql import SparkSession
from datasketch import MinHash, MinHashLSH
import random
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
from pyspark.sql import functions as F  
from pyspark.sql.functions import col
import random
import pandas as pd


# Initialize SparkSession
# spark = SparkSession.builder \
#     .appName("Movie Twins Segmentation") \
#     .getOrCreate()

spark = SparkSession.builder \
    .appName("Movie Twins Segmentation") \
    .config("spark.shuffle.partitions", 352 * 1.5) \
    .config("spark.dynamicAllocation.enabled", True) \
    .config("spark.dynamicAllocation.initialExecutors", 30) \
    .config("spark.dynamicAllocation.minExecutors", 20) \
    .config("spark.dynamicAllocation.maxExecutors", 67) \
    .config("spark.executor.memory", "12g") \
    .config("spark.executor.cores", 5) \
    .config("spark.executor.instances", 67) \
    .config("spark.driver.memory", "24g") \
    .config("spark.sql.adaptive.enabled", True) \
    .config("spark.executor.memoryOverhead", "4g") \
    .config("spark.driver.memoryOverhead", "4g") \
    .config("spark.speculation", True) \
    .getOrCreate()

# Load the ratings data
ratings_csv = "hdfs:/user/ms14845_nyu_edu/target_latest/ratings.csv"
df = spark.read.csv(ratings_csv, header=True, inferSchema=True)

# Group users by the set of movies they have rated
user_movie_sets = df.groupBy('userId').agg(F.collect_set('movieId').alias('movieIds')).collect()
user_movie_sets = {row['userId']: row['movieIds'] for row in user_movie_sets}

# ratings_csv = 'ml-latest/ratings.csv'
# df = pd.read_csv(ratings_csv, sep=',', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
# user_movie_sets = df.groupby('user_id')['movie_id'].apply(set).to_dict()


# Group users by the set of ratings they have given
user_ratings = df.groupBy('userId').agg(F.collect_set('rating').alias('ratings')).collect()
user_ratings = {row['userId']: row['ratings'] for row in user_ratings}

# Define the number of permutations for MinHash
num_perm = 128

# Create MinHash objects for each user's movie set
minhashes = {}
for user, movies in tqdm(user_movie_sets.items()):
    m = MinHash(num_perm=num_perm)
    for movie in movies:
        m.update(str(movie).encode('utf8'))
    minhashes[user] = m

# Create an LSH index
lsh = MinHashLSH(threshold=0.15, num_perm=num_perm)
for user_id, minhash in tqdm(minhashes.items()):
    lsh.insert(user_id, minhash)

# Find and display similar users (movie twins)
similar_users = {}
pairs = []
max_counts = 0
for user_id in tqdm(user_movie_sets):
    result = lsh.query(minhashes[user_id])
    # similar_users[user_id] = result
    current_cnt = 0
    for candidate in result:
        if user_id < int(candidate):
            jaccard = minhashes[user_id].jaccard(minhashes[candidate])
            pairs.append((user_id, candidate, jaccard))
            if jaccard >= 1.:
                max_counts += 1
                current_cnt += 1
            if current_cnt >= 1: break
    if max_counts > 100:
        break

# Sort pairs and take the top 100
top_100_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:100]
print('Printing the top 100 most similar user IDs')
for i, (user_a, user_b, jaccardsim) in enumerate(top_100_pairs):
    print(f"Pair {i}: User_A - {user_a}, User_B - {user_b}, Jaccard Similarity - {jaccardsim}")
# print(top_100_pairs)

correlations_top_100 = 0.
for item in top_100_pairs:
    correlations_top_100 += item[2]
correlations_top_100 = correlations_top_100/100


def calc_corr(user_pairs, user_movies, user_ratings):
    correlation = []

    for user1, user2 in user_pairs:
        movie_1 = list(user_movies[user1])
        movie_2 = list(user_movies[user2])

        ratings_1 = list(user_ratings[user1])
        ratings_2 = list(user_ratings[user2])

        length = min(len(movie_1), len(movie_2))

        rating_1_dict = dict(zip(movie_1, ratings_1))
        rating_2_dict = dict(zip(movie_2, ratings_2))

        common_movies = set(rating_1_dict.keys()).intersection(rating_2_dict.keys())

        if len(common_movies) <= 1:
            # correlation.append(0)
            continue

        common_ratings = [(rating_1_dict[movie], rating_2_dict[movie]) for movie in common_movies]

        set1, set2 = set([rating[0] for rating in common_ratings]), set([rating[1] for rating in common_ratings])
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        jaccardsim = len(intersection)/len(union)
        correlation.append(jaccardsim)

        if len(correlation) >= 100: break

    return correlation


print(f'Average correlation (top 100 similar pairs): {correlations_top_100:.2f}')

all_users = df.select('userId').distinct().rdd.map(lambda x: x[0]).collect()
random_pairs = []
while len(random_pairs) <= 10000:
    user1 = random.choice(all_users)
    user2 = random.choice(all_users)
    if user1 < user2:
        random_pairs.append((user1, user2))    
random_pairs = tuple(set(random_pairs))

# We calculate correlation only if there are common movies between the randomly picked users
correlations_random_100 = calc_corr(random_pairs, user_movie_sets, user_ratings)
correlations_random_100 = [corr for corr in correlations_random_100 if not np.isnan(corr)]

print(f'Average correlation (random 100 pairs): {np.mean(correlations_random_100):.2f}')

# Stop SparkSession
spark.stop()

