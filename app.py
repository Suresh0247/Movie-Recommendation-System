import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity



# Movie Dataset


data = {
    "Movie": [
        "Avengers",
        "Iron Man",
        "Thor",
        "Batman",
        "Superman",
        "Notebook"
    ],

    "Genre": [
        "Action Superhero",
        "Action Superhero",
        "Action Fantasy",
        "Action Dark",
        "Action Hero",
        "Romance Drama"
    ]
}

df = pd.DataFrame(data)



# Convert Text to Numbers


cv = CountVectorizer()

matrix = cv.fit_transform(df["Genre"])



# Similarity Score


similarity = cosine_similarity(matrix)



# Recommendation Function


def recommend_movie(movie_name):

    movie_index = df[df["Movie"] == movie_name].index[0]

    scores = list(enumerate(similarity[movie_index]))

    sorted_movies = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommended_movies = []

    for movie in sorted_movies[1:4]:
        recommended_movies.append(
            df.iloc[movie[0]]["Movie"]
        )

    return recommended_movies



# Streamlit UI


st.title(" Movie Recommendation System")

st.write(
    "Get movie recommendations based on similar genres."
)



# Select Movie


selected_movie = st.selectbox(
    "Choose a movie",
    df["Movie"]
)



# Recommend Button


if st.button("Recommend Movies"):

    recommendations = recommend_movie(selected_movie)

    st.subheader("Recommended Movies")

    for movie in recommendations:
        st.write( movie)
