import streamlit as st
import streamlit.components.v1 as components

from src.data_loader import load_everything
from src.recommender import RecommenderSession
from src.components import search_and_select_ui, recommendation_card, inject_card_css

st.set_page_config(page_title="Track Recommender", layout="centered")

inject_card_css()

st.title("🎵 Hybrid Music Recommender")
st.caption("Pick songs you like, rate each batch, and discover new tracks.")

st.markdown('<div id="top"></div>', unsafe_allow_html=True)

if st.session_state.get("scroll_to_top", False):
    components.html(
        """
        <script>
            const anchor = window.parent.document.getElementById("top");
            if (anchor) {
                anchor.scrollIntoView({behavior: "smooth", block: "start"});
            }
        </script>
        """,
        height=0
    )
    st.session_state.scroll_to_top = False

recommender_df, feature_matrix, knn_model, config = load_everything()

if "session" not in st.session_state:
    st.session_state.session = None
if "current_recs" not in st.session_state:
    st.session_state.current_recs = []

# feedback weights + optional genre filters as done in notebook 14 
with st.sidebar:
    st.header("Your session")

    if st.session_state.session is None:
        st.caption("Choose starting songs to begin.")
    else:
        session = st.session_state.session
        st.metric("Seed songs", len(session.seed_idxs))
        st.metric("Liked", len(session.liked_idxs))
        st.metric("Skipped", len(session.neutral_idxs))
        st.metric("Disliked", len(session.disliked_idxs))

    st.divider()

    with st.expander("Advanced recommendation settings"):
        st.caption(
            f"Model weights — Spotify: {config['spotify_weight']}, "
            f"Lyrics: {config['lyrics_weight']}, "
            f"PCA Audio: {config['pca_audio_weight']}"
        )

        liked_weight = st.slider("Like weight", 0.0, 2.0, 1.0, 0.1)
        disliked_weight = st.slider("Dislike weight", 0.0, 2.0, 0.5, 0.1)

        require_liked_genre = st.checkbox(
            "Only show tracks sharing a genre with your likes",
            value=False
        )

        remove_disliked_genre = st.checkbox(
            "Hide tracks sharing a genre with your dislikes",
            value=False
        )

    if st.session_state.session is not None:
        st.session_state.session.liked_weight = liked_weight
        st.session_state.session.disliked_weight = disliked_weight

# seed selection phase
if st.session_state.session is None:
    st.subheader("Pick at least 1 song, up to 10")
    st.caption("Search for songs, preview them, and add the versions you actually like.")
    seed_idxs = search_and_select_ui(recommender_df, max_seeds=10)

    if st.button("Generate first batch ▶", disabled=len(seed_idxs) < 1, type="primary"):
        session = RecommenderSession(
            knn_model, feature_matrix, recommender_df,
            liked_weight=liked_weight, disliked_weight=disliked_weight,
        )
        session.set_seeds(seed_idxs)
        st.session_state.session = session
        st.session_state.current_recs = session.recommend(
            k=5,
            require_liked_genre=require_liked_genre,
            remove_disliked_genre=remove_disliked_genre,
        )
        st.rerun()

# recommendations and feedback
else:
    session = st.session_state.session

    st.subheader("Recommendation batch")
    st.caption("Mark these tracks, then request the next batch.")

    for idx in st.session_state.current_recs:
        recommendation_card(idx, recommender_df, session)
        st.divider()

    st.divider()

col_next, col_reset = st.columns([2, 1])

with col_next:
    if st.button("Generate next batch ▶", type="primary", use_container_width=True):
        st.session_state.current_recs = session.recommend(
            k=5,
            require_liked_genre=require_liked_genre,
            remove_disliked_genre=remove_disliked_genre,
        )
        st.session_state.scroll_to_top = True
        st.rerun()

with col_reset:
    if st.button("New session", use_container_width=True):
        st.session_state.session = None
        st.session_state.current_recs = []
        st.session_state.seed_idxs = []
        st.session_state.scroll_to_top = True
        st.rerun()
