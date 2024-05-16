# CineMatch: Big Data Movie Recommendation Project

## Overview

The CineMatch project is dedicated to harnessing the power of big data analytics to enhance the movie-watching experience through personalized recommendations and customer segmentation. Leveraging the extensive MovieLens dataset, the project aims to tackle two fundamental challenges:

1. **Personalized Movie Recommendations**: With an ever-expanding library of films, users often face decision paralysis when selecting what to watch. By implementing collaborative filtering algorithms, the project seeks to provide tailored movie recommendations based on users' historical preferences, enhancing user satisfaction and engagement.

2. **Customer Segmentation**: Understanding user behavior and preferences is crucial for platform optimization and targeted marketing. Through customer segmentation, the project endeavors to identify groups of users with similar movie-watching styles. By employing innovative techniques like MinHash-based algorithms, the project aims to cluster users based on their movie preferences, allowing for more precise targeting of content and promotions.

By addressing these challenges, the CineMatch project endeavors to revolutionize the movie recommendation landscape, providing users with a personalized and enriching cinematic experience while empowering platforms with actionable insights for improved user engagement and satisfaction.

## Dataset Overview

The MovieLens dataset used in the CineMatch project is a substantial collection of movie ratings provided by users. The dataset contains information on a vast number of movies and users, enabling robust analysis and modeling for personalized movie recommendations and customer segmentation.

- **Number of Users**: The dataset encompasses a large user base, consisting of approximately 330,000 individuals who have rated movies. This diverse user population provides valuable insights into user preferences and behaviors.

- **Number of Movies**: The dataset includes an extensive catalog of movies, totaling 86,000 titles. Spanning various genres, release years, and popularity levels, this rich movie collection offers ample opportunities for recommending diverse and engaging content to users.

## Project Structure

- **movie_twins_segmentation.py**: This script implements customer segmentation using the MinHash-based algorithm. It calculates the similarity between users based on their movie-watching behavior and identifies pairs of users with similar preferences.

- **popularity_recommendation.py**: This script implements a baseline popularity recommendation model. It recommends movies based on their overall popularity, without considering individual user preferences.

- **data_split.py**: This script is used to partition the MovieLens dataset into training, validation, and test sets. It ensures the availability of reliable data for model training and evaluation.
  
- **als_movie_recommender.py**: This script implements a collaborative-filtering based recommendation system using the Alternating Least Squares (ALS) algorithm. It learns latent factors for users and items from the movie ratings data and generates personalized recommendations.
  
## Usage

To use the scripts provided in this repository, follow these steps:

1. Clone this repository to your machine.
2. Obtain the MovieLens dataset from [https://grouplens.org/datasets/movielens/](https://grouplens.org/datasets/movielens/).
3. Extract the downloaded dataset files to a folder.
4. Update the dataset paths in the Python files to point to the location where you extracted the MovieLens dataset.
5. Execute the scripts on respective environments (e.g., HDFS, Spark) according to individual files.
