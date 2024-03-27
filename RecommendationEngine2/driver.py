import pandas as pd
import tkinter as tk
from tkinter import Listbox, StringVar, Entry, Label, Button

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
    # if selection:
    #     selected_movie.set(movies[selection[0]])
    #     print("Selected movie:", selected_movie.get())  # For demonstration, print to console

def enter(*args):
    listbox.pack_forget()
    answerList.bindtags([answerList, app, "all"])
    answerList.insert(tk.END, 'This is where we add our results. This text currently represents one element')
    answerList.pack(fill=tk.BOTH, expand=True)

def reset(*args):
    listbox.pack(fill=tk.BOTH, expand=True)
    listbox.bind('<<ListboxSelect>>', select_movie)
    answerList.delete(0, tk.END)
    answerList.pack_forget()

app = tk.Tk()
app.title("Movie Search and Select")

answerList = Listbox(app)

search_var = StringVar()
search_var.trace("w", update_listbox)  # Trace changes to the search_var
selected_movie = StringVar()  # Variable to hold the selected movie

# Search box
search_label = Label(app, text="Search:")
search_label.pack()
search_entry = Entry(app, textvariable=search_var)
search_entry.pack()

enterButton = Button(app, text="Enter", fg='black', command=enter)
enterButton.pack(expand=False)

resetButton = Button(app, text="Reset", foreground='red', command=reset)
resetButton.pack()

# Listbox for displaying movies
listbox = Listbox(app)
listbox.pack(fill=tk.BOTH, expand=True)
listbox.bind('<<ListboxSelect>>', select_movie)  # Bind selection event

# Initialize listbox with all movies
update_listbox()

app.mainloop()

# After the GUI is closed, you can access the selected_movie variable
print("Last selected movie:", selected_movie.get())
