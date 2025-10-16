# Spotify Recommender

A music recommendation engine that generates personalized playlists using audio features from millions of tracks.  
Built in Python with **pandas** and **scikit-learn**, designed for scalability and extensibility.

## 🚀 Web App MVP (October 2025)

The recommender now runs as a Streamlit web app.  
Paste any Spotify playlist URL and instantly generate personalized song recommendations.

**Features:**
- Full Spotify API integration  
- Album art + similarity ranking display  
- Clean interactive interface built with Streamlit  

**Run locally:**
pip install -r webapp/requirements.txt
streamlit run webapp/streamlit_app.py

## 🚀 Features
- Merge and deduplicate multiple Spotify datasets (millions of rows).
- Normalize and compare audio features across tracks.
- Generate top-N recommendations for any Spotify playlist.
- CLI interface for quick testing and usage.

## 📋 Prerequisites
- Python 3.9+
- A Spotify Developer account + API credentials in `.env`

## 🛠️ Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/<your-username>/spotify-recommender.git
cd spotify-recommender
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
```

## 🎧 Usage
Run the CLI to generate recommendations:

```bash
python cli.py --playlist "https://open.spotify.com/playlist/4bvPBOdMcU0dVJQqP86upR" --top_n 20
```

Example output:

```bash
[cache] Using cached merged_997451acc9b54b36.parquet
Top 20 recommended tracks:
1. Artist – Track
2. Artist – Track
...
```

## 📂 Project Structure
```bash
spotify-recommender/
│── cli.py                  # CLI entry point
│── recommender/            # Recommendation logic
│── utils/                  # Dataset merging, preprocessing
│── data/                   # (ignored) Raw and processed datasets
│── requirements.txt
│── README.md
```

## 🔮 Roadmap
- Web app interface for playlist generation

- Support for multiple user profiles

- Mood / energy sliders for finer control

- Deployment on cloud platform

## ⚖️ License
This project is for educational and portfolio purposes.