# Spotify Recommendation Project

Solution Engineering Python Project

## Project Overview

This project focuses on building a music recommendation system using Spotify-related datasets, audio features, lyrics information, and playlist metadata.

The project explores how numerical audio characteristics and similarity-based methods can be used to recommend songs with similar musical profiles.

Current project components include:

* exploratory data analysis (EDA)
* feature preprocessing
* similarity analysis
* PCA visualization
* baseline recommendation modeling

---

# Repository Structure

```text id="9p7o2x"
spotify_project/
│   .gitignore
│   LICENSE
│   README.md
│   requirements.txt
│
├───app
│
├───config
│   │   paths.py
│   │   __init__.py
│
├───data
│   │   spotify.duckdb
│   │
│   ├───interim
│   │       spotify.duckdb
│   │
│   ├───processed
│   │       playlist_audio_features.csv
│   │       playlist_features_scaled.csv
│   │       recommender_sample.csv
│   │       scaled_audio_features.csv
│   │
│   └───raw
│       └───SpotGenTrack
│           ├───Data Sources
│           │       spotify_albums.csv
│           │       spotify_artists.csv
│           │       spotify_tracks.csv
│           │
│           └───Features Extracted
│                   low_level_audio_features.csv
│                   lyrics_features.csv
│
├───notebooks
│   │   00_duckdb_setup.ipynb
│   │   01_data_overview.ipynb
│   │   02_tracks_eda.ipynb
│   │   03_artists_eda.ipynb
│   │   04_albums_eda.ipynb
│   │   05_audio_features_eda.ipynb
│   │   06_lyrics_features_eda.ipynb
│   │   07_feature_engineering.ipynb
│   │   08_similarity_analysis.ipynb
│   │   09_playlist_eda.ipynb
│   │   10_pca_analysis.ipynb
│   │   11_baseline_recommender.ipynb
│   │   99_py3_presentation.ipynb
│   │
│   └───deepnote_archive
│           Feature Engineering and Recommendation.ipynb
│           Py3 Presentation.ipynb
│           Unzip and Load Data.ipynb
│
├───reports
│       99_py3_presentation.html
│
└───src
        create_duckdb.py
```

---

# Dataset Setup

The datasets are not included in the repository because of file size limitations.

Place the dataset files into:

```text id="86rl3z"
data/raw/SpotGenTrack/
```

using the folder structure shown above.

---

# Setup

Install required packages:

```bash id="fnhkwo"
pip install -r requirements.txt
```

Create the DuckDB database:

```bash id="rnv6p4"
python ./src/create_duckdb.py
```

This generates:

```text id="i8mrjw"
data/interim/spotify.duckdb
```
---

# Processed Data Files
| File                             | Purpose                                                    |
| -------------------------------- | ---------------------------------------------------------- |
|playlist_audio_features.csv	   | Aggregated playlist-level audio statistics|
|playlist_features_scaled.csv	   | Scaled playlist-level numerical features|
|recommender_sample.csv	           | Sample dataset used for recommender experiments|
|scaled_audio_features.csv	       | Standardized track-level audio features for similarity calculations|

---

# Notebooks

| Notebook                         | Description                                                |
| -------------------------------- | ---------------------------------------------------------- |
| 00_duckdb_setup.ipynb            | Database setup and importing CSV files into DuckDB         |
| 01_data_overview.ipynb           | Overview of datasets, table structures, and missing values |
| 02_tracks_eda.ipynb              | Exploratory analysis of track-level metadata               |
| 03_artists_eda.ipynb             | Artist-related exploration and statistics                  |
| 04_albums_eda.ipynb              | Album-related exploration                                  |
| 05_audio_features_eda.ipynb      | Analysis of extracted audio features                       |
| 06_lyrics_features_eda.ipynb     | Lyrics feature exploration                                 |
| 07_feature_engineering.ipynb     | Feature preprocessing and scaling                          |
| 08_similarity_analysis.ipynb     | Similarity calculations and cosine similarity experiments  |
| 09_playlist_eda.ipynb            | Playlist exploration and contextual analysis               |
| 10_pca_analysis.ipynb            | PCA visualization and dimensionality reduction             |
| 11_baseline_recommender.ipynb    | Baseline recommender implementation                        |
| 99_py3_presentation.ipynb        | Final presentation notebook                                |
| reports/99_py3_presentation.html | Exported HTML presentation                                 |

---

# Methods Used

The project currently explores:

* cosine similarity
* content-based recommendation
* feature scaling and normalization
* PCA (Principal Component Analysis)
* playlist-based analysis
* lyrics-based exploration

The baseline recommender uses scaled numerical audio features and cosine similarity to recommend songs with similar feature profiles.

---

# Main Technologies

* Python
* Pandas
* Scikit-learn
* DuckDB
* Librosa
* Plotly
* MLflow

---

# Current Status

Completed:

* data integration
* exploratory data analysis
* feature preprocessing
* playlist analysis
* PCA visualization
* baseline recommender implementation

Planned extensions:

* filtering non-musical content
* lyrics-based recommendation
* KNN-based recommender
* dashboard integration
* additional recommendation models