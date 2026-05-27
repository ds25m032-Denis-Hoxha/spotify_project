import duckdb
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import duckdb
from config.paths import RAW_DIR, INTERIM_DIR, DUCKDB_PATH

INTERIM_DIR.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(DUCKDB_PATH)

# Albums
con.execute(f"""
CREATE OR REPLACE TABLE albums AS
SELECT *
FROM read_csv_auto('{RAW_DIR / "SpotGenTrack/Data Sources/spotify_albums.csv"}');
""")

# Artists
con.execute(f"""
CREATE OR REPLACE TABLE artists AS
SELECT *
FROM read_csv_auto('{RAW_DIR / "SpotGenTrack/Data Sources/spotify_artists.csv"}');
""")

# Tracks
con.execute(f"""
CREATE OR REPLACE TABLE tracks AS
SELECT *
FROM read_csv_auto('{RAW_DIR / "SpotGenTrack/Data Sources/spotify_tracks.csv"}');
""")

# Audio Features
con.execute(f"""
CREATE OR REPLACE TABLE audio_features AS
SELECT *
FROM read_csv_auto('{RAW_DIR / "SpotGenTrack/Features Extracted/low_level_audio_features.csv"}');
""")

# Lyrics Features
con.execute(f"""
CREATE OR REPLACE TABLE lyrics_features AS
SELECT *
FROM read_csv_auto('{RAW_DIR / "SpotGenTrack/Features Extracted/lyrics_features.csv"}');
""")

print("DuckDB database created successfully.")

con.close()