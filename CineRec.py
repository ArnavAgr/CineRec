import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

df = pd.read_csv("movies_with_genres.csv")

genre_cols = df.columns[2:]  
features = df[genre_cols].values


# Streamlit App
st.title("CineRec : A Movie Recommendation System")
st.write("Select 1 to 4 movies you like and get 3 similar recommendations.")


selected_titles = st.multiselect(
    "Choose your favorite movies:",
    options=df['Title'].tolist(),
    max_selections=4,
    help="Start typing to search. You can select up to 4 movies."
)


if selected_titles:
    # Here we get the feature vectors for selected movies
    selected_vectors = df[df['Title'].isin(selected_titles)][genre_cols].values
    
    # Now compute average vector
    avg_vector = np.mean(selected_vectors, axis=0).reshape(1, -1)
    
    # Compute cosine similarity with all movies in the dataset
    sim_scores = cosine_similarity(avg_vector, features)[0]
    
    # Get top 3 indices excluding selected movies
    df['Similarity'] = sim_scores
    recommendations = df[~df['Title'].isin(selected_titles)]
    top_recommendations = recommendations.sort_values(by='Similarity', ascending=False).head(3)
    
    # Display recommendations
    st.subheader("Recommended Movies:")
    for idx, row in top_recommendations.iterrows():
        st.markdown(f"**{row['Title']}**")
        st.write(row['Overview'])
else:
    st.info("Please select at least one movie.")

# Run the app with: python -m streamlit run app.py

