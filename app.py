import streamlit as st

from src.data_loader import load_everything
from src.recommender import RecommenderSession
from src.components import search_and_select_ui, recommendation_card

st.set_page_config(page_title="Track Recommender", layout="centered")
st.title("🎵 Discover your next favorite track")
st.caption("Powered by the optimized hybrid recommender")

recommender_df, feature_matrix, knn_model, config = load_everything()

if "session" not in st.session_state:
    st.session_state.session = None
if "current_recs" not in st.session_state:
    st.session_state.current_recs = []

# feedback weights + optional genre filters as done in notebook 14 
with st.sidebar:
    st.header("Recommendation settings")
    st.caption(
        f"Model weights — Spotify: {config['spotify_weight']}, "
        f"Lyrics: {config['lyrics_weight']}, "
        f"PCA Audio: {config['pca_audio_weight']}"
    )

    liked_weight = st.slider("Like weight", 0.0, 2.0, 1.0, 0.1)
    disliked_weight = st.slider("Dislike weight", 0.0, 2.0, 0.5, 0.1)

    st.divider()
    require_liked_genre = st.checkbox(
        "Only show tracks sharing a genre with your likes", value=False
    )
    remove_disliked_genre = st.checkbox(
        "Hide tracks sharing a genre with your dislikes", value=False
    )

    if st.session_state.session is not None:
        st.session_state.session.liked_weight = liked_weight
        st.session_state.session.disliked_weight = disliked_weight

# seed selection phase
if st.session_state.session is None:
    st.subheader("Pick 1–10 songs you like")
    seed_idxs = search_and_select_ui(recommender_df, max_seeds=10)

    if st.button("Get recommendations ▶", disabled=len(seed_idxs) < 1):
        session = RecommenderSession(
            knn_model, feature_matrix, recommender_df,
            liked_weight=liked_weight, disliked_weight=disliked_weight,
        )
        session.set_seeds(seed_idxs)
        st.session_state.session = session
        st.session_state.current_recs = session.recommend(
            k=10,
            require_liked_genre=require_liked_genre,
            remove_disliked_genre=remove_disliked_genre,
        )
        st.rerun()

# recommendations and feedback
else:
    session = st.session_state.session
    st.subheader("Your recommendations")

    needs_refresh = False
    for idx in st.session_state.current_recs:
        if recommendation_card(idx, recommender_df, session):
            needs_refresh = True
        st.divider()

    if needs_refresh:
        st.session_state.current_recs = session.recommend(
            k=10,
            require_liked_genre=require_liked_genre,
            remove_disliked_genre=remove_disliked_genre,
        )
        st.rerun()

    if st.button("🔄 Start over"):
        st.session_state.session = None
        st.session_state.current_recs = []
        st.session_state.seed_idxs = []
        st.rerun()

    with st.expander("Session details"):
        st.write(
            f"Seeds: {len(session.seed_idxs)} | "
            f"Liked: {len(session.liked_idxs)} | "
            f"Disliked: {len(session.disliked_idxs)} | "
            f"Neutral: {len(session.neutral_idxs)}"
        )
