import numpy as np
import re


def normalize_title(title):
    title = str(title).lower().strip()

    # normalize featuring syntax
    title = title.replace("featuring", "feat.")
    title = title.replace("feat ", "feat. ")

    # remove punctuation differences
    title = title.replace("(", "")
    title = title.replace(")", "")
    title = title.replace("[", "")
    title = title.replace("]", "")

    # remove common version suffixes
    title = re.sub(
        r"\s*-\s*(live|unplugged|remastered|remaster|single|album|radio|edit|version|deluxe|acoustic).*",
        "",
        title
    )

    title = re.sub(
        r"\s+(live|unplugged|remastered|remaster|single|album|radio|edit|version|deluxe|acoustic)\s*$",
        "",
        title
    )

    # normalize whitespace
    title = re.sub(r"\s+", " ", title)

    return title.strip()

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
        self.shown_keys = set()

    def set_seeds(self, seed_idxs):
        self.seed_idxs = list(seed_idxs)
        self.shown_idxs = set(seed_idxs)

        self.shown_keys = set(
            self.recommender_df.loc[self.seed_idxs, "version_key"]
        )

    def register_feedback(self, track_idx, label):
        self.shown_idxs.add(track_idx)

        self.shown_keys.add(
            self.recommender_df.loc[track_idx, "version_key"]
        )

        if track_idx in self.liked_idxs:
            self.liked_idxs.remove(track_idx)
        if track_idx in self.disliked_idxs:
            self.disliked_idxs.remove(track_idx)
        if track_idx in self.neutral_idxs:
            self.neutral_idxs.remove(track_idx)

        if label == "like":
            self.liked_idxs.append(track_idx)
        elif label == "dislike":
            self.disliked_idxs.append(track_idx)
        else:
            self.neutral_idxs.append(track_idx)

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
            profile /= total_weight

        if disliked_vectors is not None:
            profile -= self.disliked_weight * disliked_vectors.mean(axis=0)

        return profile.reshape(1, -1)

    def _collect_genres(self, indices):
        genres = []

        for idx in indices:
            genres.extend(self.recommender_df.loc[idx, "eval_genres"])

        return set(genres)

    def recommend(
        self,
        k=5,
        require_liked_genre=False,
        remove_disliked_genre=False,
        overfetch_buffer=500,
    ):
        user_profile = self._build_user_profile()

        n_neighbors = min(
            k + len(self.shown_idxs) + overfetch_buffer,
            self.feature_matrix.shape[0],
        )

        distances, indices = self.knn_model.kneighbors(
            user_profile,
            n_neighbors=n_neighbors
        )

        candidates = self.recommender_df.iloc[indices[0]].copy()
        candidates["cosine_distance"] = distances[0]
        candidates["similarity_score"] = (1 - candidates["cosine_distance"]).clip(0, 1)

        candidates = candidates[
            ~candidates.index.isin(self.shown_idxs)
        ]

        candidates = candidates[
            ~candidates["version_key"].isin(self.shown_keys)
        ]

        # ----------------------------------------
        # Remove songs already shown under another version
        # ----------------------------------------
        shown_title_artist = {
            (
                normalize_title(self.recommender_df.loc[idx, "name"]),
                str(self.recommender_df.loc[idx, "artist_name"]).strip().lower()
            )
            for idx in self.shown_idxs
        }

        candidate_keys = list(
            zip(
                candidates["name"].apply(normalize_title),
                candidates["artist_name"].fillna("").astype(str).str.strip().str.lower()
            )
        )

        candidates = candidates[
            [key not in shown_title_artist for key in candidate_keys]
        ]

        # ----------------------------------------
        # Remove duplicates inside THIS batch
        # ----------------------------------------
        candidates["core_title"] = candidates["name"].apply(normalize_title)

        candidates["core_artist"] = (
            candidates["artist_name"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
        )

        candidates = candidates.drop_duplicates(
            subset=["core_title", "core_artist"],
            keep="first"
        )

        # ----------------------------------------
        # THEN apply genre filters
        # ----------------------------------------

        if require_liked_genre and self.liked_idxs:
            liked_genres = self._collect_genres(self.liked_idxs + self.seed_idxs)

            candidates = candidates[
                candidates["eval_genres"].apply(
                    lambda g: len(set(g) & liked_genres) > 0
                )
            ]

        if remove_disliked_genre and self.disliked_idxs:
            disliked_genres = self._collect_genres(self.disliked_idxs)

            candidates = candidates[
                candidates["eval_genres"].apply(
                    lambda g: len(set(g) & disliked_genres) == 0
                )
            ]

        candidates = candidates.drop_duplicates(
            subset=["version_key"],
            keep="first"
        )

        top_candidates = candidates.head(k)

        self.recommendation_scores = (
            top_candidates["similarity_score"].to_dict()
        )

        recs = top_candidates.index.tolist()

        self.shown_idxs.update(recs)

        self.shown_keys.update(
            self.recommender_df.loc[recs, "version_key"]
        )

        return recs