
import streamlit as st
from pathlib import Path
import sys
import json
import ast

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


st.set_page_config(
    page_title="Spotify Recommender Demo",
    page_icon="🎵",
    layout="wide"
)

st.title("Spotify Music Recommender")
st.write(
    "Search for a song, get recommendations, like or dislike them, "
    "and generate personalized recommendations."
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
processed_dir = PROJECT_ROOT / "data" / "processed"
models_dir = processed_dir / "models"


@st.cache_data
def load_data():
    tracks = pd.read_parquet(processed_dir / "tracks_with_predicted_genres.parquet")
    audio_features = pd.read_parquet(processed_dir / "audio_features_clean.parquet")
    lyrics_features = pd.read_parquet(processed_dir / "lyrics_features_valid_clean.parquet")

    with open(models_dir / "optimized_recommender_config.json", "r") as f:
        config = json.load(f)

    pca10_audio_features = joblib.load(models_dir / "pca10_audio_features.joblib")

    return tracks, audio_features, lyrics_features, config, pca10_audio_features


tracks, audio_features, lyrics_features, config, pca10_audio_features = load_data()

st.success("Data loaded successfully.")

st.write("Tracks:", len(tracks))
st.write("Audio features:", len(audio_features))
st.write("Lyrics features:", len(lyrics_features))
