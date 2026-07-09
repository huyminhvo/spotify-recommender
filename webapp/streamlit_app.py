import logging
from html import escape

import streamlit as st
from interface import (
    AppError,
    add_recommendations_to_spotify,
    format_artist_names,
    get_recommendations,
    load_catalog_bundle,
    setting_scale_to_adjustment,
)

from utils.spotify_auth import (
    create_oauth_state,
    create_user_oauth,
    decode_oauth_state,
    get_public_spotify_client,
    get_spotify_config,
    get_user_spotify_client,
)

logger = logging.getLogger(__name__)

# force Streamlit to allow custom styling overrides
st.set_page_config(
    page_title="Spotify Recommender", layout="wide", initial_sidebar_state="collapsed"
)


@st.cache_resource(show_spinner="Loading recommendation catalog...")
def get_cached_catalog_bundle():
    """Share the read-only catalog handle across Streamlit sessions."""
    return load_catalog_bundle()


try:
    # Streamlit raises when no secrets.toml exists. That is normal for local
    # development, where get_spotify_config falls back to .env.
    try:
        streamlit_secrets = st.secrets.to_dict()
    except FileNotFoundError:
        streamlit_secrets = {}
    spotify_config = get_spotify_config(streamlit_secrets)
except Exception:
    spotify_config = None
    logger.exception("Spotify configuration failed")
    st.error("Spotify is temporarily unavailable because its configuration could not be loaded.")

browser_binding = str(st.context.headers.get("User-Agent", ""))

# Spotify redirects back to this exact Streamlit URL with an authorization code.
# The signed state survives Streamlit replacing its WebSocket session during the
# external round trip.
oauth_code = st.query_params.get("code")
oauth_error = st.query_params.get("error")
oauth_error_description = st.query_params.get("error_description")
callback_state = st.query_params.get("state")
if oauth_error:
    logger.warning(
        "Spotify authorization was denied: error=%r description=%r",
        oauth_error,
        oauth_error_description,
    )
    st.error("Spotify authorization was denied. Please try connecting your account again.")
    st.session_state.pop("spotify_add_pending", None)
    st.query_params.clear()
elif oauth_code and spotify_config:
    try:
        pending_request = decode_oauth_state(spotify_config, callback_state or "", browser_binding)
        oauth, cache = create_user_oauth(spotify_config)
        oauth.get_access_token(oauth_code, check_cache=False, as_dict=False)
        st.session_state.spotify_token_info = cache.get_cached_token()
        for key in (
            "playlist_url",
            "top_n",
            "energy_setting",
            "mood_setting",
            "dance_setting",
            "acoustic_setting",
        ):
            if key in pending_request:
                st.session_state[key] = pending_request[key]
        if pending_request.get("action") == "recommend":
            st.session_state.spotify_recommend_pending = True
        st.success("Spotify account connected.")
    except ValueError:
        logger.warning("Spotify authorization state was rejected", exc_info=True)
        st.error("Spotify authorization could not be verified. Please try connecting again.")
    except Exception:
        logger.exception("Spotify authorization failed")
        st.error("Spotify authorization failed. Please try connecting again.")
    st.query_params.clear()


def get_spotify_authorize_url(config, pending_request):
    """Build a Spotify authorization URL with a signed pending request."""
    state = create_oauth_state(config, browser_binding, pending_request)
    oauth, _ = create_user_oauth(config)
    return oauth.get_authorize_url(state=state)


def render_spotify_authorize_button(config, pending_request, label="Get Recommendations"):
    """Render a user-clicked link styled like the main action button.

    Spotify refuses to render inside Streamlit Cloud's app frame. A meta refresh
    from inside the Streamlit document can therefore land the user on Chrome's
    "accounts.spotify.com refused to connect" error page. A normal user-clicked
    link avoids that frame restriction and still redirects back to this app.
    """
    authorize_url = get_spotify_authorize_url(config, pending_request)
    st.markdown(
        f"""
        <a
            href="{escape(authorize_url, quote=True)}"
            target="_top"
            rel="noopener noreferrer"
            style="
                align-items: center;
                background-color: transparent;
                border: 1px solid rgba(250, 250, 250, 0.2);
                border-radius: 0.5rem;
                color: #ffffff !important;
                display: inline-flex;
                font-size: 1rem;
                font-weight: 600;
                min-height: 2.75rem;
                padding: 0.5rem 1rem;
                text-decoration: none !important;
            "
        >
            {escape(label)}
        </a>
        """,
        unsafe_allow_html=True,
    )


def redirect_to_spotify(config, pending_request):
    """Ask the user to continue to Spotify as a top-level navigation."""
    authorize_url = get_spotify_authorize_url(config, pending_request)
    st.info("Connect your Spotify account to continue.")
    st.link_button("Continue to Spotify", authorize_url, type="primary")
    st.caption(
        "After approving access on Spotify, you'll be sent back here automatically."
    )
    st.stop()


for widget_key, default_value in {
    "playlist_url": "",
    "top_n": 10,
    "energy_setting": 5.5,
    "mood_setting": 5.5,
    "dance_setting": 5.5,
    "acoustic_setting": 5.5,
    "seen_recommendation_ids": set(),
}.items():
    st.session_state.setdefault(widget_key, default_value)


st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #fafafa;
    }
    [data-testid="stHeader"] {background: rgba(0,0,0,0);}
    [data-testid="stToolbar"] {right: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# === Custom CSS overrides ===
st.markdown(
    """
<style>
/* Global app background */
body {
    background-color: #0e1117;
    color: #fafafa;
}

/* Card container */
div[data-testid="column"] {
    background: #1a1d23;
    padding: 1rem;
    border-radius: 1rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

/* Hover effect */
div[data-testid="column"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 14px rgba(0,0,0,0.4);
}

/* Buttons / links */
a {
    color: #1DB954 !important; /* Spotify green */
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
/* Match input labels to description color */
label, .stSlider label, .stTextInput label {
    color: #b3b3b3 !important;
    font-weight: 500;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Apply font globally */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# title
st.markdown("<h1 style='color:#1DB954; '>Spotify Recommender</h1>", unsafe_allow_html=True)

# description
st.markdown(
    """
    <div style='font-size:17px; color:#b3b3b3; margin-bottom:8px;'>
        Paste a Spotify playlist link below to generate personalized song recommendations.
    </div>
    """,
    unsafe_allow_html=True,
)

# spacer between description and input
st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)


playlist_url = st.text_input(
    "Playlist URL",
    placeholder="https://open.spotify.com/playlist/...",
    key="playlist_url",
)
top_n = st.slider("Number of recommendations", min_value=5, max_value=25, key="top_n")
with st.expander("Recommendation settings"):
    slider_options = {
        "min_value": 1.0,
        "max_value": 10.0,
        "step": 0.5,
        "format": "%.1f",
        "help": "1 lowers this trait, 5.5 keeps it neutral, and 10 raises it.",
    }
    energy_setting = st.slider("Energy", key="energy_setting", **slider_options)
    mood_setting = st.slider("Mood / positivity", key="mood_setting", **slider_options)
    dance_setting = st.slider("Danceability", key="dance_setting", **slider_options)
    acoustic_setting = st.slider("Acousticness", key="acoustic_setting", **slider_options)

adjustments = {
    "energy": setting_scale_to_adjustment(energy_setting),
    "valence": setting_scale_to_adjustment(mood_setting),
    "danceability": setting_scale_to_adjustment(dance_setting),
    "acousticness": setting_scale_to_adjustment(acoustic_setting),
}
recommend_request = {
    "action": "recommend",
    "playlist_url": playlist_url,
    "top_n": top_n,
    "energy_setting": energy_setting,
    "mood_setting": mood_setting,
    "dance_setting": dance_setting,
    "acoustic_setting": acoustic_setting,
}

if st.session_state.get("spotify_token_info"):
    go = st.button("Get Recommendations")
else:
    go = False
    if playlist_url.strip() and spotify_config:
        render_spotify_authorize_button(spotify_config, recommend_request)
    elif st.button("Get Recommendations"):
        if not playlist_url.strip():
            st.warning("Please enter a playlist URL.")
        elif not spotify_config:
            st.error(
                "Spotify is temporarily unavailable because its configuration could not be loaded."
            )

if go:
    st.session_state.spotify_recommend_pending = True

if st.session_state.get("spotify_recommend_pending"):
    if not playlist_url.strip():
        st.warning("Please enter a playlist URL.")
        st.session_state.spotify_recommend_pending = False
    elif not st.session_state.get("spotify_token_info"):
        if spotify_config:
            redirect_to_spotify(spotify_config, recommend_request)
    else:
        st.session_state.spotify_recommend_pending = False
        user_sp, token_cache = get_user_spotify_client(
            spotify_config, st.session_state.spotify_token_info
        )
        with st.spinner("Fetching recommendations..."):
            try:
                recs = get_recommendations(
                    playlist_url,
                    top_n=top_n,
                    adjustments=adjustments,
                    sp=user_sp,
                    public_sp=get_public_spotify_client(spotify_config),
                    catalog_bundle=get_cached_catalog_bundle(),
                    exclude_spotify_ids=st.session_state.seen_recommendation_ids,
                )
                st.session_state.spotify_token_info = token_cache.get_cached_token()
                st.session_state.recs = recs  # persist recs across reruns
                st.session_state.seen_recommendation_ids.update(
                    recs["spotify_id"].dropna().astype(str)
                )

                if recs.empty:
                    st.error("No recommendations found.")
                else:
                    st.success(f"Top {len(recs)} recommendations:")
                    if len(recs) < top_n:
                        st.warning(
                            "The catalog ran out of unseen tracks before reaching the "
                            "requested count."
                        )
            except AppError as e:
                st.session_state.spotify_token_info = token_cache.get_cached_token()
                logger.warning(
                    "Recommendation request failed: %s",
                    e.detail or e.user_message,
                    exc_info=True,
                )
                st.error(e.user_message)
            except Exception:
                st.session_state.spotify_token_info = token_cache.get_cached_token()
                logger.exception("Unexpected recommendation failure")
                st.error("Something went wrong while generating recommendations. Please try again.")

if "recs" in st.session_state and not st.session_state.recs.empty:
    if st.button("Add These Songs to Spotify Playlist"):
        st.session_state.spotify_add_pending = True

    if st.session_state.get("spotify_add_pending"):
        token_info = st.session_state.get("spotify_token_info")
        if not token_info:
            if spotify_config:
                redirect_to_spotify(spotify_config, {"action": "add"})
        else:
            st.session_state.spotify_add_pending = False
            user_sp, token_cache = get_user_spotify_client(spotify_config, token_info)
            # API calls may refresh the token; keep the refreshed value in this
            # browser session and never write it to a shared filesystem cache.
            try:
                with st.spinner("Creating playlist in your Spotify account..."):
                    playlist_url = add_recommendations_to_spotify(st.session_state.recs, sp=user_sp)
                st.session_state.spotify_token_info = token_cache.get_cached_token()
                st.success(f"Playlist created! [Open on Spotify]({playlist_url})")
            except AppError as e:
                st.session_state.spotify_token_info = token_cache.get_cached_token()
                logger.warning(
                    "Playlist creation failed: %s",
                    e.detail or e.user_message,
                    exc_info=True,
                )
                st.error(e.user_message)
            except Exception:
                st.session_state.spotify_token_info = token_cache.get_cached_token()
                logger.exception("Unexpected playlist creation failure")
                st.error("Something went wrong while creating the playlist. Please try again.")

if "recs" in st.session_state and not st.session_state.recs.empty:
    recs = st.session_state.recs

    cols_per_row = 3
    for i in range(0, len(recs), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j >= len(recs):
                break
            row = recs.iloc[i + j]
            with col:
                if row.get("album_art_url"):
                    st.image(row["album_art_url"], use_container_width=True)
                title = row.get("title_raw", "Unknown title")
                artists_str = format_artist_names(row.get("artists_raw"))
                st.markdown(f"**{title}**  \n{artists_str}")
                track_url = f"https://open.spotify.com/track/{row['spotify_id']}"
                st.markdown(f"[Listen on Spotify]({track_url})", unsafe_allow_html=True)
                reason = row.get("recommendation_reason")
                if reason:
                    st.caption(reason)
                score = row.get("score")
                if score is not None:
                    st.caption(f"Recommendation score: {score:.4f}")
