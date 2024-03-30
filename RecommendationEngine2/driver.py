import re

import numpy as np
import Levenshtein
import pandas as pd
import tkinter as tk
from tkinter import Listbox, StringVar, Entry, Label, Button, ttk

from scipy.sparse import hstack
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)


def euclidean_distance(base_case_year: int, comparator_year: int):
    return abs(base_case_year - comparator_year)

def cosine_similarity_func(baseOverview: str, compareOverview: str):
    baseOverview = baseOverview.lower()
    compareOverview = compareOverview.lower()
    print(f"Base: \n{baseOverview}\nCompare: \n{compareOverview}")
    tfidfMatrix = TfidfVectorizer().fit_transform((baseOverview, compareOverview))
    results = cosine_similarity(tfidfMatrix[0], tfidfMatrix[1])
    return results[0][0]

def cleanid(id: str):
    id = str(id)
    if id == "0":
        return 0
    id = id.replace("tt", "")
    if id.startswith("0"):
        while id.startswith("0"):
            id = id[1:]
    return int(id)

def loadAndCleanDesc():
    descriptions = pd.read_csv('../movies_description.csv')
    descriptions.dropna(subset=['imdb_id'], inplace=True)
    descriptions['imdbId'] = descriptions['imdb_id'].map(lambda x: cleanid(x))
    descriptions = descriptions[descriptions['imdbId'] != 0]
    descriptions = descriptions.sort_values(by=['imdbId'], ascending=True)
    return descriptions

def loadMoviesWithDescriptions():
    df = pd.read_csv('../movies.csv')
    df = df.sort_values(by=['imdbId'], ascending=True)
    descriptions = loadAndCleanDesc()
    cdf = df.merge(descriptions, how='inner', on='imdbId')
    cdf = cdf.sort_values(by='movieId', ascending=True)
    return cdf


movies = loadMoviesWithDescriptions()

def compSearch(title: str, compare: str):
    title = title.lower()
    compare = compare.lower()
    if title.startswith(compare):
        return True
    else:
        return False

def compareTitle(base: str, compare: str):
    """
    :param base: the base title (e.g., 'x' in lambda x function)
    :param compare: the compare title (e.g., title of selected movie)
    :return: fraction (similar / total)

    Example: \n
    df['JaccardWordSimilarity'] = df['title'].map(lambda x: compareTitle(x, SelectedMovieTitle))
    """
    base = base.lower()
    compare = compare.lower()

    baseWords = base.split(" ")[:-1]
    compareWords = compare.split(" ")[:-1]
    badwords = ['the', 'and', 'of', 'le', 'in', 'vs.', 'to', 'a', 'on']
    baseWords = set([x for x in baseWords if x not in badwords])
    compareWords = set([x for x in compareWords if x not in badwords])

    numerator = len(baseWords.intersection(compareWords))
    denominator = len(baseWords.union(compareWords))
    if(denominator == 0):
        denominator = 1
    return float(numerator) / float(denominator)

# Function to update the listbox based on search query
def update_listbox(*args):
    search_term = search_var.get().lower()
    listbox.delete(0, tk.END)  # Clear listbox
    sdf = pd.DataFrame()
    sdf['title'] = movies['title']
    sdf['startswith'] = sdf['title'].map(lambda x: compSearch(x, search_term))
    sdf = sdf[sdf['startswith'] == True]

    matches = sdf['title'].tolist()
    for match in matches:
        listbox.insert(tk.END, match)


selection = pd.DataFrame
# Function to handle movie selection
def select_movie(*args):
    global selection
    selection = movies[movies['title'] == listbox.get(listbox.curselection())]

def enter(*args):
    listbox.pack_forget()
    print(selection)
    answerList.bindtags([answerList, app, "all"])

    k_value = int(k_spinbox.get())
    selected_title = selection['title'].iloc[0]

    cluster_title = cluster_by_title.get() == 1
    cluster_genres = cluster_by_genres.get() == 1

    global movies

    clustered_movies = pd.DataFrame()

    if cluster_title and not cluster_genres:
        clustered_movies = cluster_movies_by_title(movies, selected_title, k_value)
    elif not cluster_title and cluster_genres:
        clustered_movies = cluster_movies_by_genre(movies, selected_title, k_value)
    elif cluster_title and cluster_genres:
        clustered_movies = cluster_movies_by_title_and_genre(movies, selected_title, k_value)
    else:
        answerList.insert(tk.END, "Please select at least one option for clustering (Title or Genres).")
    #PUT CODE HERE FOR FILTERING, YOU CAN PASS IN clustered_movies
    # if filter_by_description.get() == 1:
    #     clustered_movies = cosine(clustered_movies, 5)
    print(clustered_movies) # for testing, delete it later

    filtered = filter_movies(clustered_movies)
    for index, row in filtered.iterrows():
        answerList.insert(tk.END, row['title'])
    answerList.pack(fill=tk.BOTH, expand=True)

def reset(*args):
    listbox.pack(fill=tk.BOTH, expand=True)
    listbox.bind('<<ListboxSelect>>', select_movie)
    answerList.delete(0, tk.END)
    answerList.pack_forget()

#CLUSTERING STUFF

# convert genres to one-hot encoding
def preprocess_genres(movies):
    genres = movies['genres'].str.get_dummies(sep='|')
    return genres


# One-hot encoded genres
preprocessed_data = preprocess_genres(movies)


# Cluster movies based on genres
def cluster_movies_by_genre(movies_df, selected_movie, k):
    # One-hot encode the genres
    genre_matrix = movies_df['genres'].str.get_dummies(sep='|')
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    genre_clusters = kmeans.fit_predict(genre_matrix)

    # Find the cluster of the selected movie
    selected_movie_index = movies_df.index[movies_df['title'] == selected_movie].tolist()[0]
    selected_movie_cluster = genre_clusters[selected_movie_index]

    cluster_movies = movies_df.copy()
    cluster_movies['genre_cluster'] = genre_clusters
    # Find other movies in the same cluster
    #cluster_movies = movies_df.iloc[genre_clusters == selected_movie_cluster]
    cluster_movies = cluster_movies[cluster_movies['genre_cluster'] == selected_movie_cluster]
    return cluster_movies


# def cluster_movies_by_title(movies_df, selected_movie, k):
#     # Remove years from the titles
#     movies_df['title_processed'] = movies_df['title'].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
#
#     # use tfdif
#     tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['title_processed'])
#
#     # kmeans clustering
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     title_clusters = kmeans.fit_predict(tfidf_matrix)
#
#     # find the index of the selected movie
#     selected_movie_index = movies_df.index[movies_df['title'] == selected_movie].tolist()[0]
#     selected_movie_cluster = title_clusters[selected_movie_index]
#
#     # return movies from the same cluster
#     cluster_movies = movies_df.iloc[title_clusters == selected_movie_cluster].copy()
#     # dropping the title_processed column for the output
#     cluster_movies.drop('title_processed', axis=1, inplace=True)
#
#     return cluster_movies

def cluster_movies_by_title(movies_df, selected_movie, k):
    # Copy the dataframe and add a new column for processed titles
    movies_df_copy = movies_df.copy()
    movies_df_copy['processed_title'] = movies_df_copy['title'].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

    # Extract TF-IDF features from the processed titles
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df_copy['processed_title'])

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    title_clusters = kmeans.fit_predict(tfidf_matrix)
    movies_df_copy['cluster'] = title_clusters

    # Adjust selected_movie to match the preprocessing and find its cluster
    selected_movie_cleaned = re.sub(r"\(\d{4}\)", "", selected_movie).strip()
    selected_cluster = movies_df_copy[movies_df_copy['processed_title'] == selected_movie_cleaned]['cluster'].iloc[0]

    # Return movies in the same cluster without altering the original title
    cluster_movies = movies_df_copy[movies_df_copy['cluster'] == selected_cluster]

    return cluster_movies.drop(['processed_title', 'cluster'], axis=1)

from sklearn.preprocessing import normalize


def cluster_movies_by_title_and_genre(movies_df, selected_movie, k):
    # Copy the dataframe and add a new column for processed titles
    movies_df_copy = movies_df.copy()
    movies_df_copy['processed_title'] = movies_df_copy['title'].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

    # Extract TF-IDF features from processed titles
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df_copy['processed_title'])

    # One-hot encode genres
    genre_matrix = movies_df['genres'].str.get_dummies(sep='|')

    # Combine features: TF-IDF features and one-hot encoded genre features
    combined_features = hstack([tfidf_matrix, genre_matrix])

    # Apply k-means clustering on the combined features
    kmeans = KMeans(n_clusters=k, random_state=42)
    combined_clusters = kmeans.fit_predict(combined_features)
    movies_df_copy['combined_cluster'] = combined_clusters

    # Adjust selected_movie to match the preprocessing and find its cluster
    selected_movie_cleaned = re.sub(r"\(\d{4}\)", "", selected_movie).strip()
    selected_cluster = movies_df_copy[movies_df_copy['processed_title'] == selected_movie_cleaned]['combined_cluster'].iloc[0]

    # Find other movies in the same combined cluster
    cluster_movies = movies_df_copy[movies_df_copy['combined_cluster'] == selected_cluster]

    return cluster_movies.drop(['processed_title', 'combined_cluster'], axis=1)


app = tk.Tk()
app.title("Movie Search and Select")

# This is the k-value for clustering
k_label = tk.Label(app, text="k:")
k_label.pack()

k_spinbox = ttk.Spinbox(app, from_=3, to=100, width=5)
k_spinbox.pack()

cluster_by_title = tk.IntVar(value=0)
cluster_by_genres = tk.IntVar(value=0)

# Create checkboxes for Title/Genre for clustering
title_checkbox = tk.Checkbutton(app, text="Title", variable=cluster_by_title)
genres_checkbox = tk.Checkbutton(app, text="Genres", variable=cluster_by_genres)

title_checkbox.pack()
genres_checkbox.pack()

filters_label = tk.Label(app, text="Filters")
filters_label.pack()

# Checkboxes for selecting filters
filter_by_title = tk.IntVar(value=0)
filter_by_description = tk.IntVar(value=0)
filter_by_year = tk.IntVar(value=0)

# Entry variables for percentages
title_percentage_var = tk.StringVar(value="0")
description_percentage_var = tk.StringVar(value="0")
year_percentage_var = tk.StringVar(value="0")

# Create and pack checkboxes with entry boxes for percentages
title_filter_frame = tk.Frame(app)
tk.Checkbutton(title_filter_frame, text="Title", variable=filter_by_title).pack(side=tk.LEFT)
tk.Entry(title_filter_frame, textvariable=title_percentage_var, width=5).pack(side=tk.LEFT)
title_filter_frame.pack()

description_filter_frame = tk.Frame(app)
tk.Checkbutton(description_filter_frame, text="Description", variable=filter_by_description).pack(side=tk.LEFT)
tk.Entry(description_filter_frame, textvariable=description_percentage_var, width=5).pack(side=tk.LEFT)
description_filter_frame.pack()

year_filter_frame = tk.Frame(app)
tk.Checkbutton(year_filter_frame, text="Year", variable=filter_by_year).pack(side=tk.LEFT)
tk.Entry(year_filter_frame, textvariable=year_percentage_var, width=5).pack(side=tk.LEFT)
year_filter_frame.pack()

answerList = tk.Listbox(app)

search_var = tk.StringVar()
search_var.trace("w", lambda name, index, mode, sv=search_var: update_listbox(sv))  # Adjusted for lambda
selected_movie = tk.StringVar()

search_label = tk.Label(app, text="Search:")
search_label.pack()

search_entry = tk.Entry(app, textvariable=search_var)
search_entry.pack()

enterButton = tk.Button(app, text="Enter", fg='black', command=enter)
enterButton.pack(expand=False)

resetButton = tk.Button(app, text="Reset", foreground='red', command=reset)
resetButton.pack()

listbox = tk.Listbox(app)
listbox.pack(fill=tk.BOTH, expand=True)
listbox.bind('<<ListboxSelect>>', select_movie)

update_listbox()  # Initially populate the listbox


# Filter Movies
# Cosine Similarity      (Description)
def cosine(df: pd.DataFrame, cosWeight):
    df['cosine'] = df['overview'].map(lambda x: cosine_similarity_func(str(x), str(selection['overview'].values[0])))
    sorted_df = df.sort_values(by='cosine', ascending=False)
    return sorted_df.head(cosWeight)


# Levenshtein Distance   (Title)
def levenshtein(df, levenWeight):
    # Preprocess to remove the year in parentheses from the selected movie title for comparison
    base_case_title = selection.rstrip()[:-6]  # Remove " (xxxx)" from the end

    # Function to preprocess titles in the dataframe to remove year for Levenshtein distance calculation
    def remove_year_from_title(title):
        return title.rstrip()[:-6]  # Remove " (xxxx)" from the end of each title

    # Calculating Levenshtein distance using titles without years
    df['levenshtein'] = df['title'].map(lambda x: Levenshtein.distance(remove_year_from_title(x), base_case_title))

    # Sorting by Levenshtein distance to get the closest matches (most similar first)
    sorted_df = df.sort_values(by='levenshtein', ascending=True)

    # Returning the top levenWeight matches, including their original titles with years
    return sorted_df.head(levenWeight)


# Euclidean Distance     (Year)
def euclidean(df, euclidWeight):
    df['year'] = df['title'].str.strip().str[-5:-1]
    df['year'] = pd.to_numeric(df['year'], errors='coerce').dropna()
    base_case = df.loc[selection['imdbId']]
    df['euclidean'] = df['year'].map(lambda x: euclidean_distance(float(base_case['year']), float(x)))
    sorted_df = df.sort_values(by='euclidean', ascending=False)

    return sorted_df.head(euclidWeight)

def filter_movies(df):
    choose = len(df)

    cosWeight = choose - int(choose * (float(description_percentage_var.get()) / 100)) + 3
    if cosWeight < 1:
        cosWeight = 1
    cosine(cosWeight)

    levenWeight = choose - int(choose * (float(title_percentage_var.get()) / 100)) + 3
    if levenWeight < 1:
        levenWeight = 1
    levenshtein(df, levenWeight)

    euclidWeight = choose - int(choose * (float(year_percentage_var.get()) / 100)) + 3
    if euclidWeight < 1:
        euclidWeight = 1
    df = euclidean(df, euclidWeight)

    # This is so that if the selected movie is here, it is removed
    df = df[df['title'] != selection]

    return df

app.mainloop()

print("Last selected movie:", selected_movie.get())
