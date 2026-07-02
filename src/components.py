import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from rapidfuzz import process, fuzz


def inject_card_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: #f7f7f5;
            color: #1f1f1f;
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e6e6e6;
        }

        h1, h2, h3 {
            color: #191414 !important;
        }

        p, span, label, div {
            color: #191414;
        }

        [data-testid="stCaptionContainer"] {
            color: #6f6f6f !important;
        }

        div[data-testid="stTextInput"] input {
            background-color: #ffffff;
            color: #191414;
            border: 1px solid #dddddd;
            border-radius: 12px;
        }

        div[data-testid="stTextInput"] input:focus {
            border: 1px solid #1DB954;
            box-shadow: 0 0 0 1px #1DB954;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #e4e4e4;
            border-radius: 18px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.06);
        }

        button[kind="primary"] {
            background: #1DB954 !important;
            color: white !important;
            border-radius: 999px !important;
            border: none !important;
            font-weight: 700 !important;
        }

        button {
            border-radius: 999px !important;
        }

        hr {
            border-color: #e5e5e5 !important;
        }

        iframe {
            border-radius: 12px;
        }

        div[data-testid="stExpander"] {
            background: #ffffff;
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def spotify_embed(spotify_id, height=80):
    if not spotify_id:
        st.caption("No Spotify preview available for this track.")
        return

    embed_url = f"https://open.spotify.com/embed/track/{spotify_id}?utm_source=generator"
    components.iframe(embed_url, height=height, scrolling=False)

def format_track_caption(row, genres):
    album = row.get("album_name", "Unknown Album")
    year = row.get("release_year", "")

    album_year = album
    if year:
        album_year = f"{album} • {year}"

    return (
        f"{row.get('artist_name', 'Unknown Artist')} · "
        f"{album_year} · {genres} · popularity {int(row['popularity'])}"
    )


def search_and_select_ui(recommender_df, max_seeds=10):
    if "seed_idxs" not in st.session_state:
        st.session_state.seed_idxs = []

    if "search_reset_counter" not in st.session_state:
        st.session_state.search_reset_counter = 0

    query = st.text_input(
        "Search for a song",
        placeholder="e.g. Billie Jean",
        key=f"song_search_{st.session_state.search_reset_counter}"
    )

    if query:
        q = query.lower().strip()

        exact = recommender_df[
            recommender_df["name"].str.lower().str.strip() == q
        ].copy()
        exact["search_rank"] = 0

        starts = recommender_df[
            recommender_df["name"].str.lower().str.startswith(q, na=False)
        ].copy()
        starts["search_rank"] = 1

        contains = recommender_df[
            recommender_df["name"].str.lower().str.contains(q, na=False)
        ].copy()
        contains["search_rank"] = 2

        choices = recommender_df["name"].tolist()
        fuzzy_results = process.extract(
            query,
            choices,
            limit=20,
            scorer=fuzz.WRatio,
            score_cutoff=90
        )

        fuzzy_indices = [
            recommender_df.index[pos_idx]
            for _, _, pos_idx in fuzzy_results
        ]

        fuzzy = recommender_df.loc[fuzzy_indices].copy()
        fuzzy["search_rank"] = 3

        direct_results = pd.concat(
            [exact, starts, contains],
            axis=0
        ).drop_duplicates(subset=["version_key"])

        if len(direct_results) >= 5:
            search_df = direct_results
        else:
            search_df = pd.concat(
                [direct_results, fuzzy],
                axis=0
            )

        search_df = search_df.drop_duplicates(subset=["version_key"])

        query_words = [word for word in q.split() if len(word) > 2]

        if query_words:
            search_text = (
                search_df["name"].fillna("").str.lower()
                + " "
                + search_df.get("artist_name", "").fillna("").str.lower()
                + " "
                + search_df.get("album_name", "").fillna("").str.lower()
            )

            search_df = search_df[
                search_df["search_rank"].lt(3)
                | search_text.apply(lambda text: all(word in text for word in query_words))
            ]

        search_df = (
            search_df
            .drop_duplicates(subset=["version_key"])
            .sort_values(["search_rank", "popularity"], ascending=[True, False])
            .head(10)
        )

        if not search_df.empty:
            st.markdown("### Search results")
            st.caption("Preview a track and add the correct version to your starting songs.")

        for idx, row in search_df.iterrows():
            genres = ", ".join(row["eval_genres"][:2]) if row["eval_genres"] else "—"

            with st.container(border=True):
                col1, col2 = st.columns([5, 1])

                with col1:
                    st.markdown(f"**{row['name']}**")
                    st.caption(format_track_caption(row, genres))

                    with st.expander("Preview"):
                        spotify_embed(row.get("spotify_id"), height=152)

                with col2:
                    already_added = idx in st.session_state.seed_idxs
                    disabled = already_added or len(st.session_state.seed_idxs) >= max_seeds
                    label = "Added ✓" if already_added else "Add"

                    if st.button(
                        label,
                        key=f"add_{idx}",
                        disabled=disabled,
                        use_container_width=True
                    ):
                        st.session_state.seed_idxs.append(idx)
                        st.session_state.search_reset_counter += 1
                        st.rerun()

    if st.session_state.seed_idxs:
        st.divider()
        st.markdown(f"### Selected starting songs ({len(st.session_state.seed_idxs)}/{max_seeds})")
        st.caption("These songs define the initial taste profile for the recommender.")

        for idx in st.session_state.seed_idxs:
            row = recommender_df.loc[idx]
            seed_genres = ", ".join(row["eval_genres"][:2]) if row["eval_genres"] else "—"

            with st.container(border=True):
                col1, col2 = st.columns([5, 1])

                with col1:
                    st.markdown(f"**{row['name']}**")
                    st.caption(format_track_caption(row, seed_genres))

                    with st.expander("Preview selected song"):
                        spotify_embed(row.get("spotify_id"), height=152)

                with col2:
                    if st.button("Remove", key=f"remove_{idx}", use_container_width=True):
                        st.session_state.seed_idxs.remove(idx)
                        st.rerun()

    return st.session_state.seed_idxs


def recommendation_card(idx, recommender_df, session):
    row = recommender_df.loc[idx]
    genres = ", ".join(row["eval_genres"][:3]) if row["eval_genres"] else "—"

    already_rated = (
        idx in session.liked_idxs
        or idx in session.disliked_idxs
        or idx in session.neutral_idxs
    )

    with st.container(border=True):
        st.markdown(f"### {row['name']}")
        st.caption(format_track_caption(row, f"🎵 {genres}"))

        if idx in session.liked_idxs:
            st.success("✓ Added to your taste profile")
        elif idx in session.disliked_idxs:
            st.error("✓ Reduced in future recommendations")
        elif idx in session.neutral_idxs:
            st.info("✓ Skipped for now")

        if not already_rated:
            spotify_embed(row.get("spotify_id"), height=152)
        else:
            st.caption("Preview hidden after feedback.")

        col1, col2, col3 = st.columns(3)
        clicked = None

        if col1.button("❤️ Like", key=f"like_{idx}", disabled=already_rated):
            clicked = "like"
        if col2.button("😐 Skip", key=f"neutral_{idx}", disabled=already_rated):
            clicked = "neutral"
        if col3.button("❌ Dislike", key=f"dislike_{idx}", disabled=already_rated):
            clicked = "dislike"

        if clicked:
            session.register_feedback(idx, clicked)
            st.rerun()

    return False