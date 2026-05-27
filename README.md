# Spotify Recommendation Project

Solution Engineering Python Project

## Project Structure

```text
spotify_project/
├── app/
├── config/
├── data/
│   ├── interim/
│   │   └── spotify.duckdb
│   ├── processed/
│   └── raw/
│       └── SpotGenTrack/
│           ├── Data Sources/
│           │   ├── spotify_albums.csv
│           │   ├── spotify_artists.csv
│           │   └── spotify_tracks.csv
│           │
│           └── Features Extracted/
│               ├── low_level_audio_features.csv
│               └── lyrics_features.csv
│
├── notebooks/
├── reports/
└── src/
```

---

## Dataset Setup

The datasets are NOT included in the GitHub repository.

Each one of the members must manually place the dataset files into:

```text
data/raw/SpotGenTrack/
```

with the exact folder structure shown above.

---

## Initial Setup

Install required packages:

```bash
pip install -r requirements.txt
```

Create the DuckDB database:

```bash
python ./src/create_duckdb.py
```

This will generate:

```text
data/interim/spotify.duckdb
```

---

## Main Technologies

- Python
- DuckDB
- Pandas
- Scikit-learn
- Librosa
- Plotly
- MLflow