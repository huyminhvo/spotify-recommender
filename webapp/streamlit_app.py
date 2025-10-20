#link to ari playlist: https://open.spotify.com/playlist/7BH80QsOwuL7h92CG9ap9w?si=e1748481e0a742e0
import streamlit as st
import pandas as pd
from interface import get_recommendations  # adjust import if needed

# Force Streamlit to allow custom styling overrides
st.set_page_config(page_title="Spotify Recommender", page_icon="🎧", layout="wide", initial_sidebar_state="collapsed")

# Force dark theme for consistent visual style
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

# --- Custom CSS overrides ---
st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Match input labels to description color */
label, .stSlider label, .stTextInput label {
    color: #b3b3b3 !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Apply font globally */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)



# Title
st.markdown("<h1 style='color:#1DB954; '>Spotify Recommender</h1>", unsafe_allow_html=True)

# Description (tight to title)
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


playlist_url = st.text_input("Playlist URL", placeholder="https://open.spotify.com/playlist/...")
top_n = st.slider("Number of recommendations", min_value=5, max_value=25, value=10)
go = st.button("Get Recommendations")

if go:
    if not playlist_url.strip():
        st.warning("Please enter a playlist URL.")
    else:
        with st.spinner("Fetching recommendations..."):
            try:
                recs = get_recommendations(playlist_url, top_n=top_n)

                if recs.empty:
                    st.error("No recommendations found.")
                else:
                    st.success(f"Top {len(recs)} recommendations:")

                    cols_per_row = 3
                    for i in range(0, len(recs), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j >= len(recs):
                                break
                            row = recs.iloc[i + j]

                            with col:
                                # album image
                                if row.get("album_art_url"):
                                    st.image(row["album_art_url"], use_container_width=True)

                                # title + artist
                                title = row.get("title_raw", "Unknown title")
                                artists = row.get("artists_raw", [])
                                if isinstance(artists, list):
                                    artists_str = ", ".join(artists)
                                else:
                                    artists_str = str(artists)
                                st.markdown(f"**{title}**  \n{artists_str}")

                                # Spotify link
                                track_url = f"https://open.spotify.com/track/{row['spotify_id']}"
                                st.markdown(
                                    f"[🎧 Listen on Spotify]({track_url})",
                                    unsafe_allow_html=True,
                                )

                                # optional similarity metric
                                sim = row.get("similarity")
                                if sim is not None:
                                    st.caption(f"Recommendation score: {sim:.4f}")

            except Exception as e:
                st.error(f"Error: {e}")
