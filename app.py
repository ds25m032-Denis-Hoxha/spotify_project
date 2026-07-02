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

try:
    recommender_df, feature_matrix, knn_model, config = load_everything()
except FileNotFoundError as exc:
    st.error("The dashboard is working, but the recommender data files were not found.")

    st.write("Please make sure the processed data and model files exist in:")

    st.code(
        """
data/processed/
data/processed/models/
        """,
        language="text"
    )

    st.write("Expected main files:")

    st.code(
        """
tracks_with_predicted_genres.parquet
audio_features_clean.parquet
lyrics_features_valid_clean.parquet
models/optimized_recommender_config.json
models/pca10_audio_features.joblib
        """,
        language="text"
    )

    st.info("Run the project notebooks up to Notebook 14, or copy the processed data folder into the project.")

    with st.expander("Technical details"):
        st.code(str(exc), language="text")

    st.stop()

except Exception as exc:
    st.error("The dashboard started, but something went wrong while loading the recommender data.")

    with st.expander("Technical details"):
        st.code(str(exc), language="text")

    st.stop()


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
        st.markdown(f"🎵 **Seeds:** {len(session.seed_idxs)}")
        st.markdown(f"❤️ **Liked:** {len(session.liked_idxs)}")
        st.markdown(f"😐 **Skipped:** {len(session.neutral_idxs)}")
        st.markdown(f"❌ **Disliked:** {len(session.disliked_idxs)}")

    st.divider()

    with st.expander("Advanced recommendation settings"):
        st.markdown("### Model info")
        st.markdown("**Active model:** Optimized Hybrid Recommender")

        st.caption(
            "This model combines three feature groups to calculate song similarity."
        )

        st.write(f"Track audio features: **{int(config['spotify_weight'] * 100)}%**")
        st.write(f"Lyrics features: **{int(config['lyrics_weight'] * 100)}%**")
        st.write(f"PCA audio features: **{int(config['pca_audio_weight'] * 100)}%**")

        st.divider()

        st.markdown("### Feedback controls")
        st.caption("These settings control how strongly your feedback changes the next batch.")

        liked_weight = st.slider("Like weight", 0.0, 2.0, 1.0, 0.1)
        disliked_weight = st.slider("Dislike weight", 0.0, 2.0, 0.5, 0.1)

        st.divider()

        st.markdown("### Genre filters *(Experimental)*")

        st.caption(
            "These filters are still experimental because many songs belong to multiple genres. "
            "Using them may reduce recommendation diversity."
        )
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

    st.subheader("Recommended for You")
    st.caption("Rate these tracks, then generate your next personalized batch.")

    for i, idx in enumerate(st.session_state.current_recs, start=1):
        recommendation_card(
            idx,
            recommender_df,
            session,
            i
        )
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

st.divider()
st.caption(
    "Hybrid recommender using track audio features, lyrics features, and PCA-reduced audio features."
)