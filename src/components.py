import ast
import html
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rapidfuzz import process, fuzz


# ---------- Display helpers ----------

ARTIST_COLUMNS = [
    "artist_names",
    "artists_names",
    "artist_name",
    "artists_name",
    "artists",
    "artist",
]

ALBUM_COLUMNS = ["album_name", "album", "album_title"]

RELEASE_COLUMNS = ["release_date", "album_release_date", "release_year"]


def _is_empty(value: Any) -> bool:
    """Return True for missing/empty scalar or list-like display values."""
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in {"", "[]", "nan", "None"}:
        return True
    if isinstance(value, (list, tuple, set)) and len(value) == 0:
        return True
    return False


def _format_list_like(value: Any) -> str:
    """Convert artist/list fields into a clean comma-separated string."""
    if _is_empty(value):
        return ""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple, set)):
                    return ", ".join(str(x) for x in parsed if not _is_empty(x))
            except Exception:
                return text.strip("[]'")
        return text
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(x) for x in value if not _is_empty(x))
    if hasattr(value, "tolist"):
        return _format_list_like(value.tolist())
    return str(value)


def artist_display(row: pd.Series) -> str:
    """Best-effort artist name display across possible dataset column names."""
    for col in ARTIST_COLUMNS:
        if col in row.index:
            value = _format_list_like(row.get(col))
            if value:
                return value
    return "Unknown artist"


def _first_existing(row: pd.Series, columns: list[str], default: str = "") -> str:
    for col in columns:
        if col in row.index:
            value = _format_list_like(row.get(col))
            if value:
                return value
    return default


def _duration_display(row: pd.Series) -> str:
    if "duration_ms" not in row.index or _is_empty(row.get("duration_ms")):
        return ""
    try:
        seconds = int(row.get("duration_ms")) // 1000
        return f"{seconds // 60}:{seconds % 60:02d}"
    except Exception:
        return ""


def _track_version_details(row: pd.Series) -> str:
    """Details that help users distinguish covers, remasters, albums, edits, etc."""
    parts = []
    album = _first_existing(row, ALBUM_COLUMNS)
    release = _first_existing(row, RELEASE_COLUMNS)
    duration = _duration_display(row)

    if album:
        parts.append(album)
    if release:
        parts.append(str(release)[:10])
    if duration:
        parts.append(duration)
    if "popularity" in row.index and not _is_empty(row.get("popularity")):
        try:
            parts.append(f"popularity {int(row.get('popularity'))}")
        except Exception:
            pass

    return " · ".join(parts)


def track_heading(row: pd.Series) -> str:
    return f"{row['name']} — {artist_display(row)}"


def _safe_text(value: Any) -> str:
    return html.escape(str(value), quote=False)


def compact_track_header(row: pd.Series, meta: str | None = None):
    """Render a compact, consistent song header for seed and recommendation cards."""
    title = _safe_text(row.get("name", "Unknown track"))
    artist = _safe_text(artist_display(row))
    meta_html = _safe_text(meta or "")
    meta_block = f'<div class="track-meta">{meta_html}</div>' if meta_html else ""
    st.markdown(
        f"""
        <div class="track-title">{title}</div>
        <div class="track-artist">{artist}</div>
        {meta_block}
        """,
        unsafe_allow_html=True,
    )


def spotify_embed(spotify_id, preview_url=None, height=80):
    """Embed Spotify if a track id is available; otherwise use the 30s preview URL if present."""
    if spotify_id:
        safe_id = html.escape(str(spotify_id), quote=True)
        embed_url = f"https://open.spotify.com/embed/track/{safe_id}?utm_source=generator"
        components.iframe(embed_url, height=height)
        return

    if preview_url and not _is_empty(preview_url):
        st.audio(preview_url)
        return

    st.caption("No Spotify preview available for this track.")


def track_preview(row: pd.Series, height=80):
    """Preview helper for full track rows."""
    spotify_embed(row.get("spotify_id"), row.get("preview_url"), height=height)


# ---------- Search and seed selection ----------

def _search_choices(recommender_df: pd.DataFrame) -> list[str]:
    """Search over title + artist + album/version info, not only the track name."""
    choices = []
    for _, row in recommender_df.iterrows():
        choices.append(
            " | ".join(
                part for part in [
                    str(row.get("name", "")),
                    artist_display(row),
                    _track_version_details(row),
                ]
                if part
            )
        )
    return choices


def search_and_select_ui(recommender_df, max_seeds=10):
    if "seed_idxs" not in st.session_state:
        st.session_state.seed_idxs = []

    query = st.text_input(
        "Search for a song",
        placeholder="e.g. Billie Jean, Michael Jackson, remaster, album version",
    )

    if query:
        cache_key = "search_choices_cache"
        if (
            cache_key not in st.session_state
            or len(st.session_state[cache_key]) != len(recommender_df)
        ):
            st.session_state[cache_key] = _search_choices(recommender_df)
        choices = st.session_state[cache_key]

        results = process.extract(
            query,
            choices,
            limit=15,
            scorer=fuzz.WRatio,
            score_cutoff=50,
        )

        if not results:
            st.info("No matching songs found. Try another title, artist, or album keyword.")

        for _, score, pos_idx in results:
            idx = recommender_df.index[pos_idx]
            row = recommender_df.loc[idx]
            genres = ", ".join(row["eval_genres"][:2]) if row["eval_genres"] else ""
            version_details = _track_version_details(row)

            with st.container():
                col1, col2 = st.columns([6, 1.15], vertical_alignment="center")
                with col1:
                    st.markdown(f"**{row['name']}** · {artist_display(row)}")
                    secondary = " · ".join(part for part in [version_details, genres] if part)
                    if secondary:
                        st.caption(secondary)
                already_added = idx in st.session_state.seed_idxs
                disabled = already_added or len(st.session_state.seed_idxs) >= max_seeds
                label = "Added ✓" if already_added else "Add"
                if col2.button(label, key=f"add_{idx}", disabled=disabled, use_container_width=True):
                    st.session_state.seed_idxs.append(idx)
                    st.rerun()

    if st.session_state.seed_idxs:
        st.markdown(f"**Selected seeds ({len(st.session_state.seed_idxs)}/{max_seeds}):**")
        for idx in list(st.session_state.seed_idxs):
            row = recommender_df.loc[idx]
            version_details = _track_version_details(row)

            with st.container(border=True):
                col1, col2 = st.columns([6, 1.35], vertical_alignment="center")
                with col1:
                    compact_track_header(row, version_details)
                if col2.button("Remove", key=f"remove_{idx}", use_container_width=True):
                    st.session_state.seed_idxs.remove(idx)
                    st.rerun()

                with st.expander("Preview", expanded=False):
                    track_preview(row, height=80)

    return st.session_state.seed_idxs


# ---------- Recommendation card ----------

def recommendation_card(idx, recommender_df, session):
    row = recommender_df.loc[idx]
    genres = ", ".join(row["eval_genres"][:3]) if row["eval_genres"] else "—"
    version_details = _track_version_details(row)
    current_feedback = session.feedback_for(idx)
    meta = " · ".join(part for part in [version_details, f"Genres: {genres}"] if part)

    with st.container(border=True):
        compact_track_header(row, meta)
        track_preview(row, height=80)

        feedback_label = current_feedback.capitalize() if current_feedback else "Not rated yet"
        st.caption(f"Feedback: **{feedback_label}**")

        col1, col2, col3 = st.columns(3, gap="small")
        clicked = None
        if col1.button("👍 Like", key=f"like_{idx}", use_container_width=True):
            clicked = "like"
        if col2.button("— Neutral", key=f"neutral_{idx}", use_container_width=True):
            clicked = "neutral"
        if col3.button("👎 Dislike", key=f"dislike_{idx}", use_container_width=True):
            clicked = "dislike"

    if clicked:
        session.register_feedback(idx, clicked)
        st.toast(f"Feedback saved: {clicked}")
        return True
    return False
