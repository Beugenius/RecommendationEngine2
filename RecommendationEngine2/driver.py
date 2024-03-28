import pandas as pd
import tkinter as tk
from tkinter import Listbox, StringVar, Entry, Label, Button, ttk

from sklearn.cluster import KMeans

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)



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
    print(cdf.head())
    return df


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

# Function to handle movie selection
def select_movie(*args):
    selection = listbox.get(listbox.curselection())
    selectedRow = movies[movies['title'] == selection]
    print(selectedRow)

def enter(selected_movie_arg):
    listbox.pack_forget()
    answerList.bindtags([answerList, app, "all"])
    answerList.insert(tk.END, 'This is where we add our results. This text currently represents one element')
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

# Jaccard Similarity for Titles
def calc_jaccard_similarity(selected_movie, movies_df):
    titles = movies_df['title'].tolist()
    title_sets = [set(title.lower().split()) for title in titles]
    selected_title_set = set(selected_movie.lower().split)


app = tk.Tk()
app.title("Movie Search and Select")

# This is the k-value for clustering
k_label = tk.Label(app, text="k:")
k_label.pack()

k_spinbox = ttk.Spinbox(app, from_=3, to=100, width=5)
k_spinbox.pack()

cluster_by_title = tk.IntVar(value=0)  # Default to checked for "Title"
cluster_by_genres = tk.IntVar(value=0)  # Default to unchecked for "Genres"

# Create checkboxes for Title/Genre for clustering
title_checkbox = tk.Checkbutton(app, text="Title", variable=cluster_by_title)
genres_checkbox = tk.Checkbutton(app, text="Genres", variable=cluster_by_genres)

title_checkbox.pack()
genres_checkbox.pack()

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

app.mainloop()

print("Last selected movie:", selected_movie.get())
