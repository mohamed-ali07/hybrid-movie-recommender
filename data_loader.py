import os
import requests
import zipfile
import io
import pandas as pd

DATA_DIR = "data"
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def download_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if os.path.exists(os.path.join(DATA_DIR, "ml-latest-small")):
        return

    print("Downloading dataset...")
    response = requests.get(DATA_URL)

    if response.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(response.content))
        z.extractall(DATA_DIR)
        print("Download complete!")

def load_data():
    download_dataset()

    movies = pd.read_csv("data/ml-latest-small/movies.csv")
    ratings = pd.read_csv("data/ml-latest-small/ratings.csv")

    return movies, ratings