import pickle
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import requests

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=aec39c1309a8c1870f3471eae5e4e724&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters

def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_idx= movies[movies['title'] == movie_name].index[0]
    # movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
    distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)
    rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
    recommend_frame = []
    # poster_frame = []
    for val in rec_movie_indices:
        movie_idx = final_dataset.iloc[val[0]]['movieId']
        idx = movies[movies['movieId'] == movie_idx].index
        recommend_frame.append(movies.iloc[idx]['title'].values[0])
        # poster_frame.append(fetch_poster(movie_idx))
    # df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
    return recommend_frame

st.header('Movie Recommender System')
movies = pickle.load(open('movies.pkl','rb'))
csr_data = pickle.load(open('csr_data.pkl','rb'))
final_dataset = pickle.load(open('final_dataset.pkl','rb'))

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendations'):
    names = get_movie_recommendation(selected_movie)
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(names[i])
            # st.image(posters[i])