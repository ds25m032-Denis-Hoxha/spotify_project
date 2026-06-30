import numpy as np


class RecommenderSession:
    def __init__(
        self,
        knn_model,
        feature_matrix,
        recommender_df,
        seed_weight=1.0,
        liked_weight=1.0,
        disliked_weight=0.5,
    ):
        self.knn_model = knn_model
        self.feature_matrix = feature_matrix
        self.recommender_df = recommender_df

        self.seed_weight = seed_weight
        self.liked_weight = liked_weight
        self.disliked_weight = disliked_weight

        self.seed_idxs = []
        self.liked_idxs = []
        self.disliked_idxs = []
        self.neutral_idxs = []
        self.shown_idxs = set()

    # Setup, called once after user picks song(s)
    def set_seeds(self, seed_idxs):
        self.seed_idxs = list(seed_idxs)
        self.shown_idxs = set(seed_idxs)

    # Feedback with like, neutral and dislike buttons
    def register_feedback(self, track_idx, label):
        self.shown_idxs.add(track_idx)
        if label == "like":
            self.liked_idxs.append(track_idx)
        elif label == "dislike":
            self.disliked_idxs.append(track_idx)
        else:
            self.neutral_idxs.append(track_idx)  # seen only, no profile shift

    # Profile construction (extended nb14 logic)
    
    def _build_user_profile(self):
        seed_vectors = self.feature_matrix[self.seed_idxs] if self.seed_idxs else None
        liked_vectors = self.feature_matrix[self.liked_idxs] if self.liked_idxs else None
        disliked_vectors = self.feature_matrix[self.disliked_idxs] if self.disliked_idxs else None

        profile = np.zeros(self.feature_matrix.shape[1])
        total_weight = 0.0

        if seed_vectors is not None:
            profile += self.seed_weight * seed_vectors.mean(axis=0)
            total_weight += self.seed_weight
        if liked_vectors is not None:
            profile += self.liked_weight * liked_vectors.mean(axis=0)
            total_weight += self.liked_weight

        if total_weight > 0:
            profile /= total_weight  # keep scale stable regardless of how many groups contributed

        if disliked_vectors is not None:
            profile -= self.disliked_weight * disliked_vectors.mean(axis=0)

        return profile.reshape(1, -1)

    # Genre helpers from notebook 14

    def _collect_genres(self, indices):
        genres = []
        for idx in indices:
            genres.extend(self.recommender_df.loc[idx, "eval_genres"])
        return set(genres)

    # Recommendation

    def recommend(
        self,
        k=10,
        require_liked_genre=False,
        remove_disliked_genre=False,
        overfetch_buffer=300,
    ):
        user_profile = self._build_user_profile()

        n_neighbors = min(
            k + len(self.shown_idxs) + overfetch_buffer,
            self.feature_matrix.shape[0],
        )
        distances, indices = self.knn_model.kneighbors(user_profile, n_neighbors=n_neighbors)

        candidates = self.recommender_df.iloc[indices[0]].copy()
        candidates["cosine_distance"] = distances[0]
        candidates = candidates[~candidates.index.isin(self.shown_idxs)]

        if require_liked_genre and self.liked_idxs:
            liked_genres = self._collect_genres(self.liked_idxs + self.seed_idxs)
            candidates = candidates[
                candidates["eval_genres"].apply(lambda g: len(set(g) & liked_genres) > 0)
            ]

        if remove_disliked_genre and self.disliked_idxs:
            disliked_genres = self._collect_genres(self.disliked_idxs)
            candidates = candidates[
                candidates["eval_genres"].apply(lambda g: len(set(g) & disliked_genres) == 0)
            ]

        candidates = candidates.drop_duplicates(subset=["name"], keep="first")

        recs = candidates.head(k).index.tolist()
        self.shown_idxs.update(recs)
        return recs
