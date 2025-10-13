import streamlit as st

st.set_page_config(page_title="Spotify Recommender", page_icon="🎧", layout="wide")

st.title("Spotify Recommender (MVP)")
st.caption("Paste a Spotify playlist URL and get recommendations.")

playlist_url = st.text_input("Playlist URL", placeholder="https://open.spotify.com/playlist/...")
go = st.button("Get Recommendations")

if go:
    st.info("Backend not wired yet — this is just the shell. Next steps will hook your CLI pipeline.")
