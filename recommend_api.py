from flask import Flask
from flask import jsonify
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask_cors import CORS
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

overall_stats = pd.read_csv('Data_train.csv', header=None)

column_names1 = ['user id', 'product id', 'rating', 'timestamp']
dataset = pd.read_csv('Data_train.csv', sep=',', header=None, names=column_names1)
refined_dataset = dataset.groupby(by=['user id', 'product id'], as_index=False).agg({"rating": "mean"})

num_users = len(refined_dataset['user id'].value_counts())
num_items = len(refined_dataset['product id'].value_counts())

rating_count_df = pd.DataFrame(refined_dataset.groupby(['rating']).size(), columns=['count'])

total_count = num_items * num_users
zero_count = total_count - refined_dataset.shape[0]

rating_count_df = rating_count_df.append(
    pd.DataFrame({'count': zero_count}, index=[0.0]),
    verify_integrity=True,
).sort_index()

print(rating_count_df)

rating_count_df['log_count'] = np.log(rating_count_df['count'])
rating_count_df = rating_count_df.reset_index().rename(columns={'index': 'rating score'})

movies_count_df = pd.DataFrame(refined_dataset.groupby('product id').size(), columns=['count'])

user_to_movie_df = refined_dataset.pivot(
    index='user id',
    columns='product id',
    values='rating').fillna(0)

user_to_movie_sparse_df = csr_matrix(user_to_movie_df.values)

knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_to_movie_sparse_df)

movies_list = user_to_movie_df.columns


def get_similar_users(user):
    ## input to this function is the user and number of top similar users you want.

    knn_input = np.asarray([user_to_movie_df.values[user - 1]])  # .reshape(1,-1)
    # knn_input = user_to_movie_df.iloc[0,:].values.reshape(1,-1)
    distances, indices = knn_model.kneighbors(knn_input, n_neighbors=5 + 1)

    print("Top", 5, "users who are very much similar to the User-", user, "are: ")
    print(" ")
    for i in range(1, len(distances[0])):
        print(i, ". User:", indices[0][i] + 1, "separated by distance of", distances[0][i])
    return indices.flatten()[1:] + 1, distances.flatten()[1:]


def test(user_id, n):
    similar_user_list, distance_list = get_similar_users(user_id)

    weightage_list = distance_list / np.sum(distance_list)
    mov_rtngs_sim_users = user_to_movie_df.values[similar_user_list]

    movies_list = user_to_movie_df.columns

    weightage_list = weightage_list[:, np.newaxis] + np.zeros(len(movies_list))

    new_rating_matrix = weightage_list * mov_rtngs_sim_users
    mean_rating_list = new_rating_matrix.sum(axis=0)

    n = min(len(mean_rating_list), n)

    first_zero_index = np.where(mean_rating_list == 0)[0][-1]
    sortd_index = np.argsort(mean_rating_list)[::-1]
    sortd_index = sortd_index[:list(sortd_index).index(first_zero_index)]
    n = min(len(sortd_index), n)
    movies_watched = list(refined_dataset[refined_dataset['user id'] == user_id]['product id'])
    filtered_movie_list = list(movies_list[sortd_index])
    count = 0
    final_movie_list = []
    for i in filtered_movie_list:
        if i not in movies_watched:
            count += 1
            final_movie_list.append(i)
        if count == n:
            break
    if count == 0:
        print(
            "There are no movies left which are not seen by the input users and seen by similar users. May be increasing the number of similar users who are to be considered may give a chance of suggesting an unseen good movie.")
        return []
    else:
        print(final_movie_list)
        return final_movie_list


@app.route('/rcm/<int:user>')
def recommend_movies(user):
    return jsonify(test(user, 9))


if __name__ == '__main__':
    app.run()
