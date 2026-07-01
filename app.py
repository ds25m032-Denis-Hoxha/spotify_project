import streamlit as st

from src.data_loader import MissingDataError, load_everything
from src.recommender import RecommenderSession
from src.components import (
    artist_display,
    recommendation_card,
    search_and_select_ui,
    track_preview,
)

st.set_page_config(page_title="Track Recommender", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.15rem;
            padding-bottom: 1.75rem;
            max-width: 1280px;
        }
        h1 {
            font-size: clamp(2.4rem, 5vw, 4.2rem) !important;
            line-height: 1.02 !important;
            letter-spacing: -0.045em;
            margin-bottom: 0.25rem !important;
        }
        h2, h3 {
            letter-spacing: -0.025em;
            margin-bottom: 0.55rem !important;
        }
        div[data-testid="stCaptionContainer"] {
            line-height: 1.35;
        }
        div[data-testid="stVerticalBlock"] > div:has(iframe) {
            margin-top: -0.15rem;
            margin-bottom: -0.45rem;
        }
        iframe {
            border-radius: 0.75rem;
        }
        .stButton button {
            min-height: 2.4rem;
            padding: 0.25rem 0.65rem;
            border-radius: 0.75rem;
            font-weight: 650;
        }
        .stTextInput input {
            border-radius: 0.85rem;
        }
        div[data-testid="stExpander"] {
            border-radius: 0.85rem;
        }
        hr {
            margin: 0.65rem 0;
        }
        .track-title {
            font-size: 1.05rem;
            font-weight: 750;
            line-height: 1.2;
            margin-bottom: 0.15rem;
        }
        .track-artist {
            color: rgba(250, 250, 250, 0.74);
            font-size: 0.96rem;
            font-weight: 500;
            line-height: 1.25;
            margin-bottom: 0.30rem;
        }
        .track-meta {
            color: rgba(250, 250, 250, 0.58);
            font-size: 0.84rem;
            line-height: 1.25;
            margin-bottom: 0.55rem;
        }
        .help-card {
            border: 1px solid rgba(250, 250, 250, 0.12);
            border-radius: 1rem;
            padding: 1.05rem 1.1rem;
            background: rgba(255, 255, 255, 0.035);
        }
        .help-card p, .help-card li {
            line-height: 1.42;
            margin-bottom: 0.4rem;
        }
        .tiny-note {
            color: rgba(250, 250, 250, 0.62);
            font-size: 0.88rem;
            line-height: 1.35;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎵 Discover your next favorite track")
try:
    recommender_df, feature_matrix, knn_model, config = load_everything()
except MissingDataError as exc:
    st.error("The dashboard is installed correctly, but no recommender data was found yet.")
    st.write("Place or generate at least one of the baseline files from the README:")
    st.code("""data/processed/recommender_sample.csv
data/processed/scaled_audio_features.csv""", language="text")
    st.write("Or generate the optimized dashboard files:")
    st.code("""data/processed/tracks_with_predicted_genres.parquet
data/processed/audio_features_clean.parquet
data/processed/lyrics_features_valid_clean.parquet
data/processed/models/optimized_recommender_config.json
data/processed/models/pca10_audio_features.joblib""", language="text")
    st.info("Based on your README, run the notebooks up to the baseline recommender or copy your existing data folder into this project.")
    with st.expander("Technical details"):
        st.code(str(exc), language="text")
    st.stop()

st.caption(f"Powered by {config.get('mode', 'the recommender')}")

if "session" not in st.session_state:
    st.session_state.session = None
if "current_recs" not in st.session_state:
    st.session_state.current_recs = []
if "seed_idxs" not in st.session_state:
    st.session_state.seed_idxs = []

# feedback weights + optional genre filters as done in notebook 14
with st.sidebar:
    st.header("Recommendation settings")
    st.caption(
        f"Mode: {config.get('mode', 'recommender')} | "
        f"Spotify/audio: {config['spotify_weight']}, "
        f"Lyrics: {config['lyrics_weight']}, "
        f"PCA Audio: {config['pca_audio_weight']}"
    )

    rec_batch_size = st.slider("Recommendations per batch", 5, 20, 10, 1)
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


def generate_recommendations(session):
    return session.recommend(
        k=rec_batch_size,
        require_liked_genre=require_liked_genre,
        remove_disliked_genre=remove_disliked_genre,
    )


# seed selection phase
if st.session_state.session is None:
    left, right = st.columns([2.35, 1], gap="large")

    with left:
        st.subheader("Pick 1–10 songs you like")
        seed_idxs = search_and_select_ui(recommender_df, max_seeds=10)

        if st.button("Generate recommendations", disabled=len(seed_idxs) < 1, type="primary"):
            session = RecommenderSession(
                knn_model,
                feature_matrix,
                recommender_df,
                liked_weight=liked_weight,
                disliked_weight=disliked_weight,
            )
            session.set_seeds(seed_idxs)
            st.session_state.session = session
            st.session_state.current_recs = generate_recommendations(session)
            st.rerun()

    with right:
        st.markdown(
            """
            <div class="help-card">
                <h3 style="margin-top:0;">How it works</h3>
                <p>Search by song, artist, album, or version.</p>
                <p>Add up to 10 seed songs and preview them with Spotify.</p>
                <p>Generate recommendations, save feedback, then refresh manually.</p>
                <div class="tiny-note">Tip: 2–4 seed songs usually create a clearer taste profile.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# recommendations and feedback
else:
    session = st.session_state.session

    header_col, action_col = st.columns([3, 1.35], gap="large")
    with header_col:
        st.subheader("Your recommendations")
        st.caption(
            "Like, neutral, and dislike are saved first. A new batch appears only when you refresh."
        )
    with action_col:
        refresh_clicked = st.button(
            "🔄 New batch from feedback",
            type="primary",
            use_container_width=True,
        )

    if refresh_clicked:
        st.session_state.current_recs = generate_recommendations(session)
        st.rerun()

    with st.expander("Your seed songs", expanded=False):
        for idx in session.seed_idxs:
            row = recommender_df.loc[idx]
            st.markdown(f"**{row['name']}** · {artist_display(row)}")
            track_preview(row, height=80)

    if not st.session_state.current_recs:
        st.warning("No recommendations found with the current filters. Try relaxing the genre filters.")
    else:
        feedback_changed = False
        for row_start in range(0, len(st.session_state.current_recs), 2):
            cols = st.columns(2, gap="medium")
            for col, idx in zip(cols, st.session_state.current_recs[row_start:row_start + 2]):
                with col:
                    if recommendation_card(idx, recommender_df, session):
                        feedback_changed = True

        if feedback_changed:
            st.rerun()

    bottom_col1, bottom_col2 = st.columns([1, 1])
    if bottom_col1.button("🔄 Start over", use_container_width=True):
        st.session_state.session = None
        st.session_state.current_recs = []
        st.session_state.seed_idxs = []
        st.rerun()

    if bottom_col2.button("Generate another batch", use_container_width=True):
        st.session_state.current_recs = generate_recommendations(session)
        st.rerun()

    with st.expander("Session details"):
        st.write(
            f"Seeds: {len(session.seed_idxs)} | "
            f"Liked: {len(session.liked_idxs)} | "
            f"Disliked: {len(session.disliked_idxs)} | "
            f"Neutral: {len(session.neutral_idxs)}"
        )
