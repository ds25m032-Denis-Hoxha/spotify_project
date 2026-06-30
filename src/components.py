import streamlit as st
import streamlit.components.v1 as components
from rapidfuzz import process, fuzz

# search box and UI to select
def search_and_select_ui(recommender_df, max_seeds=10):
    if "seed_idxs" not in st.session_state:
        st.session_state.seed_idxs = []

    query = st.text_input("Search for a song", placeholder="e.g. Billie Jean")

    if query:
        choices = recommender_df["name"].tolist()
        results = process.extract(query, choices, limit=15, scorer=fuzz.WRatio, score_cutoff=50)

        for matched_str, score, pos_idx in results:
            idx = recommender_df.index[pos_idx]
            row = recommender_df.loc[idx]
            col1, col2 = st.columns([5, 1])
            genres = ", ".join(row["eval_genres"][:2]) if row["eval_genres"] else ""
            col1.write(f"**{row['name']}**  ·  {genres}")
            already_added = idx in st.session_state.seed_idxs
            disabled = already_added or len(st.session_state.seed_idxs) >= max_seeds
            label = "Added ✓" if already_added else "Add"
            if col2.button(label, key=f"add_{idx}", disabled=disabled):
                st.session_state.seed_idxs.append(idx)
                st.rerun()

    if st.session_state.seed_idxs:
        st.markdown(f"**Selected seeds ({len(st.session_state.seed_idxs)}/{max_seeds}):**")
        for idx in st.session_state.seed_idxs:
            row = recommender_df.loc[idx]
            col1, col2 = st.columns([5, 1])
            col1.write(row["name"])
            if col2.button("Remove", key=f"remove_{idx}"):
                st.session_state.seed_idxs.remove(idx)
                st.rerun()

    return st.session_state.seed_idxs

# embedding spotify player
def spotify_embed(spotify_id, height=80):
    if not spotify_id:
        st.caption("No Spotify preview available for this track.")
        return
    embed_url = f"https://open.spotify.com/embed/track/{spotify_id}?utm_source=generator"
    components.iframe(embed_url, height=height)


# recommendation card with like/neutral/dislike buttons
def recommendation_card(idx, recommender_df, session):
    row = recommender_df.loc[idx]
    genres = ", ".join(row["eval_genres"][:3]) if row["eval_genres"] else "—"

    st.markdown(f"**{row['name']}**")
    st.caption(f"Genres: {genres}  ·  Popularity: {int(row['popularity'])}")
    spotify_embed(row.get("spotify_id"), height=80)

    col1, col2, col3 = st.columns(3)
    clicked = None
    if col1.button("👍 Like", key=f"like_{idx}"):
        clicked = "like"
    if col2.button("➖ Neutral", key=f"neutral_{idx}"):
        clicked = "neutral"
    if col3.button("👎 Dislike", key=f"dislike_{idx}"):
        clicked = "dislike"

    if clicked:
        session.register_feedback(idx, clicked)
        return True
    return False
