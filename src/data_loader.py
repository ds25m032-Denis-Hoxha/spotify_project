"""
Loading the optimized hybrid recommender artifacts produced by
notebooks/13_feature_group_optimization.ipynb and rebuilding the weighted feature matrix used in notebooks/14_interactive_recommender.ipynb.
Using notebook 14 logic for streamlit
"""

import json
import ast
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"


def _to_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, np.ndarray):
        return val.tolist()
    if pd.isna(val):
        return []
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return [val]
    return []


def _spotify_uri_to_url(uri):
    if pd.isna(uri):
        return None
    if isinstance(uri, str) and uri.startswith("spotify:track:"):
        track_id = uri.split(":")[-1]
        return f"https://open.spotify.com/track/{track_id}"
    return None


def _spotify_id_from_uri(uri):
    if pd.isna(uri):
        return None
    if isinstance(uri, str) and uri.startswith("spotify:track:"):
        return uri.split(":")[-1]
    return None

def _first_artist_id(artists_id):
    ids = _to_list(artists_id)
    return ids[0] if ids else None

def _release_year(value):
    if pd.isna(value):
        return ""
    value = str(value)
    return value[:4] if len(value) >= 4 else value

@st.cache_resource
def load_everything():
    """
    Rebuilds recommender_df, X_recommender, KNN model and caches them
    """
    tracks = pd.read_parquet(PROCESSED_DIR / "tracks_with_predicted_genres.parquet")
    audio_features = pd.read_parquet(PROCESSED_DIR / "audio_features_clean.parquet")
    lyrics_features = pd.read_parquet(PROCESSED_DIR / "lyrics_features_valid_clean.parquet")
    artists = pd.read_parquet(PROCESSED_DIR / "artists_clean.parquet")
    albums = pd.read_parquet(PROCESSED_DIR / "albums_clean.parquet")

    with open(MODELS_DIR / "optimized_recommender_config.json", "r") as f:
        config = json.load(f)

    pca10_audio_features = joblib.load(MODELS_DIR / "pca10_audio_features.joblib")

    # rebuild recommender_df as notebook 14 does 
    recommender_df = (
        tracks
        .merge(audio_features, left_on="id", right_on="track_id", how="inner")
        .merge(lyrics_features, left_on="id", right_on="track_id", how="inner",
               suffixes=("", "_lyrics"))
    )

    duplicate_track_id_cols = [
        col for col in recommender_df.columns
        if "track_id" in col and col != "track_id"
    ]
    recommender_df = recommender_df.drop(columns=duplicate_track_id_cols, errors="ignore")

    recommender_df["eval_genres"] = recommender_df["predicted_genre_list"].apply(_to_list)
    recommender_df = recommender_df[
        recommender_df["eval_genres"].str.len() > 0
    ].reset_index(drop=True)

    # display-friendly Spotify links / embed IDs
    recommender_df["spotify_url"] = recommender_df["uri"].apply(_spotify_uri_to_url)
    recommender_df["spotify_id"] = recommender_df["uri"].apply(_spotify_id_from_uri)

    artists_lookup = artists[["id", "name"]].rename(
        columns={"id": "main_artist_id", "name": "artist_name"}
    )

    recommender_df["main_artist_id"] = recommender_df["artists_id"].apply(_first_artist_id)

    recommender_df = recommender_df.merge(
        artists_lookup,
        on="main_artist_id",
        how="left"
    )

    recommender_df["artist_name"] = recommender_df["artist_name"].fillna("Unknown Artist")

    albums_lookup = albums[["id", "name", "release_date"]].rename(
        columns={
            "id": "album_id",
            "name": "album_name"
        }
    )

    recommender_df = recommender_df.merge(
        albums_lookup,
        on="album_id",
        how="left"
    )

    recommender_df["album_name"] = recommender_df["album_name"].fillna("Unknown Album")
    recommender_df["release_year"] = recommender_df["release_date"].apply(_release_year)

    recommender_df["version_key"] = (
        recommender_df["name"].str.lower().str.strip()
        + " | "
        + recommender_df["artist_name"].str.lower().str.strip()
        + " | "
        + recommender_df["album_name"].str.lower().str.strip()
        + " | "
        + recommender_df["release_year"].astype(str).str.lower().str.strip()
    )


    # align PCA audio features to the same row order/filter as recommender_df
    pca10_audio_features = pca10_audio_features.loc[recommender_df.index].reset_index(drop=True)

    # rebuild the weighted hybrid matrix as in notebook 14
    spotify_features = config["spotify_features"]
    lyrics_features_final = config["lyrics_features"]
    spotify_weight = config["spotify_weight"]
    lyrics_weight = config["lyrics_weight"]
    pca_audio_weight = config["pca_audio_weight"]

    spotify_scaled = StandardScaler().fit_transform(recommender_df[spotify_features])
    lyrics_scaled = StandardScaler().fit_transform(recommender_df[lyrics_features_final])
    pca_audio_scaled = StandardScaler().fit_transform(pca10_audio_features)

    X_recommender = np.hstack([
        spotify_scaled * spotify_weight,
        lyrics_scaled * lyrics_weight,
        pca_audio_scaled * pca_audio_weight,
    ])

    knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=51)
    knn_model.fit(X_recommender)

    return recommender_df, X_recommender, knn_model, config
