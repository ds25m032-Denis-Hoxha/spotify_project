"""
Data loading for the Streamlit dashboard.

The project exists in two possible states:

1. Optimized/hybrid recommender state:
   data/processed/tracks_with_predicted_genres.parquet
   data/processed/audio_features_clean.parquet
   data/processed/lyrics_features_valid_clean.parquet
   data/processed/models/optimized_recommender_config.json
   data/processed/models/pca10_audio_features.joblib

2. README/baseline state:
   data/processed/recommender_sample.csv
   or data/processed/scaled_audio_features.csv
   or raw spotify_tracks.csv after dataset setup.

This loader supports both so the dashboard does not crash just because the
advanced notebook artifacts have not been generated yet.
"""

import ast
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
RAW_TRACKS_PATH = DATA_DIR / "raw" / "SpotGenTrack" / "Data Sources" / "spotify_tracks.csv"
RAW_ARTISTS_PATH = DATA_DIR / "raw" / "SpotGenTrack" / "Data Sources" / "spotify_artists.csv"
RAW_ALBUMS_PATH = DATA_DIR / "raw" / "SpotGenTrack" / "Data Sources" / "spotify_albums.csv"

OPTIMIZED_FILES = {
    "tracks": PROCESSED_DIR / "tracks_with_predicted_genres.parquet",
    "audio": PROCESSED_DIR / "audio_features_clean.parquet",
    "lyrics": PROCESSED_DIR / "lyrics_features_valid_clean.parquet",
    "config": MODELS_DIR / "optimized_recommender_config.json",
    "pca": MODELS_DIR / "pca10_audio_features.joblib",
}

BASELINE_CANDIDATES = [
    PROCESSED_DIR / "recommender_sample.csv",
    PROCESSED_DIR / "scaled_audio_features.csv",
    RAW_TRACKS_PATH,
]

BASELINE_FEATURE_CANDIDATES = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
    "duration_ms",
    "popularity",
]


class MissingDataError(FileNotFoundError):
    """Raised when no usable recommender data exists in the expected project folders."""


def _to_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, np.ndarray):
        return val.tolist()
    if val is None:
        return []
    try:
        if pd.isna(val):
            return []
    except Exception:
        pass
    if isinstance(val, str):
        text = val.strip()
        if not text or text in {"[]", "nan", "None"}:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        except Exception:
            pass
        return [text]
    return []


def _spotify_uri_to_url(uri):
    if pd.isna(uri):
        return None
    if isinstance(uri, str) and uri.startswith("spotify:track:"):
        track_id = uri.split(":")[-1]
        return f"https://open.spotify.com/track/{track_id}"
    if isinstance(uri, str) and uri.startswith("https://open.spotify.com/track/"):
        return uri
    return None


def _spotify_id_from_uri(uri):
    if pd.isna(uri):
        return None
    if isinstance(uri, str) and uri.startswith("spotify:track:"):
        return uri.split(":")[-1]
    if isinstance(uri, str) and "open.spotify.com/track/" in uri:
        return uri.split("/track/")[-1].split("?")[0]
    return None


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _read_artists_table():
    """Load artist metadata if available in processed or raw data folders."""
    candidates = [
        PROCESSED_DIR / "artists_clean.parquet",
        PROCESSED_DIR / "artists.parquet",
        RAW_ARTISTS_PATH,
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                return _read_table(candidate)
            except Exception:
                continue

    return None


def _read_albums_table():
    """Load album metadata if available so duplicate versions are easier to distinguish."""
    candidates = [
        PROCESSED_DIR / "albums_clean.parquet",
        PROCESSED_DIR / "albums.parquet",
        RAW_ALBUMS_PATH,
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                return _read_table(candidate)
            except Exception:
                continue

    return None


def _artist_names_from_ids(artist_ids, artist_lookup):
    ids = _to_list(artist_ids)
    names = []
    for artist_id in ids:
        artist_id = str(artist_id).strip()
        if artist_id in artist_lookup:
            names.append(str(artist_lookup[artist_id]))
    return ", ".join(names) if names else "Unknown artist"


def _ensure_artist_names(tracks: pd.DataFrame) -> pd.DataFrame:
    """Add a display-friendly artist_names column."""
    existing_artist_cols = [
        "artist_names",
        "artists_names",
        "artist_name",
        "artists_name",
        "artists",
        "artist",
    ]
    for col in existing_artist_cols:
        if col in tracks.columns:
            tracks["artist_names"] = tracks[col]
            return tracks

    if "artists_id" not in tracks.columns:
        tracks["artist_names"] = "Unknown artist"
        return tracks

    artists = _read_artists_table()
    if artists is None or not {"id", "name"}.issubset(artists.columns):
        tracks["artist_names"] = "Unknown artist"
        return tracks

    artist_lookup = {
        str(artist_id).strip(): artist_name
        for artist_id, artist_name in zip(artists["id"], artists["name"])
        if not pd.isna(artist_id) and not pd.isna(artist_name)
    }
    tracks["artist_names"] = tracks["artists_id"].apply(
        lambda ids: _artist_names_from_ids(ids, artist_lookup)
    )
    return tracks


def _ensure_album_details(tracks: pd.DataFrame) -> pd.DataFrame:
    """Add album_name/release_date when album metadata is available."""
    if "album_name" in tracks.columns and "release_date" in tracks.columns:
        return tracks
    if "album_id" not in tracks.columns:
        return tracks

    albums = _read_albums_table()
    if albums is None or "id" not in albums.columns:
        return tracks

    album_columns = ["id"]
    rename_map = {}
    if "name" in albums.columns:
        album_columns.append("name")
        rename_map["name"] = "album_name"
    if "release_date" in albums.columns:
        album_columns.append("release_date")
    elif "album_release_date" in albums.columns:
        album_columns.append("album_release_date")
        rename_map["album_release_date"] = "release_date"

    if len(album_columns) == 1:
        return tracks

    album_small = albums[album_columns].drop_duplicates("id").rename(
        columns={"id": "album_id", **rename_map}
    )
    return tracks.merge(album_small, on="album_id", how="left")


def _ensure_common_display_columns(tracks: pd.DataFrame) -> pd.DataFrame:
    """Normalize different notebook column names into what the dashboard expects."""
    tracks = tracks.copy().reset_index(drop=True)

    if "name" not in tracks.columns:
        for col in ["track_name", "track_name_prev", "title"]:
            if col in tracks.columns:
                tracks["name"] = tracks[col]
                break
    if "name" not in tracks.columns:
        tracks["name"] = tracks.index.map(lambda i: f"Track {i + 1}")

    if "id" not in tracks.columns:
        tracks["id"] = tracks.index.map(lambda i: f"row_{i}")

    tracks = _ensure_artist_names(tracks)
    tracks = _ensure_album_details(tracks)

    if "eval_genres" not in tracks.columns:
        if "predicted_genre_list" in tracks.columns:
            tracks["eval_genres"] = tracks["predicted_genre_list"].apply(_to_list)
        elif "genre_list" in tracks.columns:
            tracks["eval_genres"] = tracks["genre_list"].apply(_to_list)
        elif "genres" in tracks.columns:
            tracks["eval_genres"] = tracks["genres"].apply(_to_list)
        else:
            tracks["eval_genres"] = [[] for _ in range(len(tracks))]

    if "spotify_url" not in tracks.columns:
        tracks["spotify_url"] = tracks["uri"].apply(_spotify_uri_to_url) if "uri" in tracks.columns else None
    if "spotify_id" not in tracks.columns:
        tracks["spotify_id"] = tracks["uri"].apply(_spotify_id_from_uri) if "uri" in tracks.columns else None
    if "preview_url" not in tracks.columns:
        tracks["preview_url"] = None

    return tracks


def _optimized_files_available() -> bool:
    return all(path.exists() for path in OPTIMIZED_FILES.values())


def _build_knn(feature_matrix: np.ndarray) -> NearestNeighbors:
    n_neighbors = min(51, feature_matrix.shape[0])
    knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    knn_model.fit(feature_matrix)
    return knn_model


def _load_optimized():
    tracks = pd.read_parquet(OPTIMIZED_FILES["tracks"])
    tracks = _ensure_common_display_columns(tracks)
    audio_features = pd.read_parquet(OPTIMIZED_FILES["audio"])
    lyrics_features = pd.read_parquet(OPTIMIZED_FILES["lyrics"])

    with open(OPTIMIZED_FILES["config"], "r", encoding="utf-8") as f:
        config = json.load(f)

    pca10_audio_features = joblib.load(OPTIMIZED_FILES["pca"])

    recommender_df = (
        tracks
        .merge(audio_features, left_on="id", right_on="track_id", how="inner")
        .merge(lyrics_features, left_on="id", right_on="track_id", how="inner", suffixes=("", "_lyrics"))
    )

    duplicate_track_id_cols = [
        col for col in recommender_df.columns
        if "track_id" in col and col != "track_id"
    ]
    recommender_df = recommender_df.drop(columns=duplicate_track_id_cols, errors="ignore")
    recommender_df = recommender_df[
        recommender_df["eval_genres"].str.len() > 0
    ].reset_index(drop=True)

    pca10_audio_features = pca10_audio_features.loc[recommender_df.index].reset_index(drop=True)

    spotify_features = config["spotify_features"]
    lyrics_features_final = config["lyrics_features"]
    spotify_weight = config["spotify_weight"]
    lyrics_weight = config["lyrics_weight"]
    pca_audio_weight = config["pca_audio_weight"]

    spotify_scaled = StandardScaler().fit_transform(recommender_df[spotify_features])
    lyrics_scaled = StandardScaler().fit_transform(recommender_df[lyrics_features_final])
    pca_audio_scaled = StandardScaler().fit_transform(pca10_audio_features)

    feature_matrix = np.hstack([
        spotify_scaled * spotify_weight,
        lyrics_scaled * lyrics_weight,
        pca_audio_scaled * pca_audio_weight,
    ])

    config["mode"] = "optimized hybrid recommender"
    knn_model = _build_knn(feature_matrix)
    return recommender_df, feature_matrix, knn_model, config


def _load_baseline():
    source_path = _first_existing(BASELINE_CANDIDATES)
    if source_path is None:
        expected = "\n".join(str(path.relative_to(PROJECT_ROOT)) for path in BASELINE_CANDIDATES)
        raise MissingDataError(
            "No recommender data found. Expected one of these files:\n" + expected
        )

    recommender_df = _read_table(source_path)
    recommender_df = _ensure_common_display_columns(recommender_df)

    available_features = [
        col for col in BASELINE_FEATURE_CANDIDATES
        if col in recommender_df.columns and pd.api.types.is_numeric_dtype(recommender_df[col])
    ]

    if not available_features:
        raise MissingDataError(
            f"Found {source_path.relative_to(PROJECT_ROOT)}, but it does not contain usable numeric audio features. "
            f"Expected some of: {', '.join(BASELINE_FEATURE_CANDIDATES)}"
        )

    clean_df = recommender_df.dropna(subset=available_features).reset_index(drop=True)
    if clean_df.empty:
        raise MissingDataError(
            f"Found {source_path.relative_to(PROJECT_ROOT)}, but all rows have missing values in the audio features."
        )

    feature_matrix = StandardScaler().fit_transform(clean_df[available_features])
    knn_model = _build_knn(feature_matrix)

    config = {
        "mode": f"baseline audio-feature recommender ({source_path.relative_to(PROJECT_ROOT)})",
        "spotify_features": available_features,
        "lyrics_features": [],
        "spotify_weight": 1.0,
        "lyrics_weight": 0.0,
        "pca_audio_weight": 0.0,
    }

    return clean_df, feature_matrix, knn_model, config


@st.cache_resource
def load_everything():
    """Return recommender_df, feature_matrix, KNN model and config for the dashboard."""
    if _optimized_files_available():
        return _load_optimized()
    return _load_baseline()
