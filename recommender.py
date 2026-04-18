import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommender:

    def __init__(self, movies, ratings):
        self.movies = movies.copy()
        self.ratings = ratings.copy()

        self.prepare_content()
        self.train_svd()

    # ---------- CONTENT ----------
    def prepare_content(self):
        self.movies['genres'] = self.movies['genres'].fillna('')
        self.movies['genres_str'] = self.movies['genres'].str.replace('|', ' ')

        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies['genres_str'])

        self.cosine_sim = cosine_similarity(self.tfidf_matrix)

        self.movie_index_map = pd.Series(self.movies.index, index=self.movies['movieId'])

    # ---------- COLLABORATIVE ----------
    def train_svd(self):
        user_movie_matrix = self.ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)

        self.user_ids = user_movie_matrix.index.tolist()
        self.movie_ids = user_movie_matrix.columns.tolist()

        svd = TruncatedSVD(n_components=50, random_state=42)
        reduced = svd.fit_transform(user_movie_matrix)

        self.reconstructed = np.dot(reduced, svd.components_)

        self.user_index_map = {uid: i for i, uid in enumerate(self.user_ids)}
        self.movie_col_map = {mid: i for i, mid in enumerate(self.movie_ids)}

    # ---------- HYBRID ----------
    def recommend(self, user_id, movie_title, cf_weight=0.6, content_weight=0.4, top_n=10):

        if movie_title not in self.movies['title'].values:
            return pd.DataFrame()

        idx = self.movies[self.movies['title'] == movie_title].index[0]

        # Content scores
        content_scores = self.cosine_sim[idx]

        # CF scores
        cf_scores_full = np.zeros(len(self.movies))

        if user_id in self.user_index_map:
            user_idx = self.user_index_map[user_id]
            user_pred = self.reconstructed[user_idx]

            for movie_id, col_idx in self.movie_col_map.items():
                if movie_id in self.movie_index_map:
                    movie_idx = self.movie_index_map[movie_id]
                    cf_scores_full[movie_idx] = user_pred[col_idx]

        # Normalize CF
        if np.max(cf_scores_full) > 0:
            cf_scores_full = (cf_scores_full - np.min(cf_scores_full)) / (
                np.max(cf_scores_full) - np.min(cf_scores_full)
            )

        # Hybrid score (DYNAMIC)
        hybrid_scores = (cf_weight * cf_scores_full) + (content_weight * content_scores)

        self.movies['hybrid_score'] = hybrid_scores
        self.movies['content_score'] = content_scores
        self.movies['cf_score'] = cf_scores_full

        result = self.movies[self.movies['title'] != movie_title]
        result = result.sort_values(by='hybrid_score', ascending=False)

        return result[['title', 'genres', 'hybrid_score', 'content_score', 'cf_score']].head(top_n)