# fastscore.schema.0: input
# fastscore.schema.1: output

# fastscore.module-attached: autorecommender
from autorecommender.models.autoencoder import load_model
from autorecommender.data import Dataset, ratings_matrix_to_list

import pandas as pd
import numpy as np
import sys

# modelop.init
def begin():
    global movies, model, rating_counts
    movies = pd.read_csv("/fastscore/datasets/movies.csv")
    rating_counts = pd.read_csv("/fastscore/datasets/rating_counts.csv").set_index(["movieid"])
    rating_counts["count"] = rating_counts["count"]/rating_counts["count"].max()
    model = load_model("/fastscore/artifacts/autorecommender.zip")
    print("MODEL LOADED")
    sys.stdout.flush()

# modelop.score
def action(ratings):
    rows = [{"userid": 0, "movieid": int(x), "rating": ratings[x]} for x in ratings]

    if len(rows) != 0:
        ratings_list = pd.DataFrame(rows).set_index(["userid", "movieid"])
        predictions = model.predict(ratings_list).transpose()
        predictions[0] = predictions[0]/predictions[0].max()
        predictions["score"] = 2*predictions[0] + rating_counts["count"]
        # drop films they've already seen
        predictions = predictions.drop([2797, 2105, 1573, 1517, 260, 1307, 1721])

        films = movies.iloc[predictions.sort_values(by="score", ascending=False).iloc[0:10].index]
        yield list(films["title"])
