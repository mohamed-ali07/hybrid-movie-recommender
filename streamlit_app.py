import streamlit as st
from data_loader import load_data
from recommender import HybridRecommender

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_data
def load_model():
    movies, ratings = load_data()
    model = HybridRecommender(movies, ratings)
    return model, movies, ratings

model, movies, ratings = load_model()

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    background: linear-gradient(90deg, #ff4b4b, #ff9966);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #cccccc;
    margin-bottom: 25px;
}

.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 18px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    transition: 0.3s;
}

.card:hover {
    transform: scale(1.05);
}

.movie-title {
    font-size: 16px;
    font-weight: bold;
    color: white;
}

.genre {
    font-size: 13px;
    color: #bbbbbb;
}

.score-badge {
    background-color: #ff4b4b;
    color: white;
    padding: 4px 10px;
    border-radius: 8px;
    font-size: 12px;
    display: inline-block;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Recommendation Control")

    cf_weight = st.slider("Collaborative Influence", 0.0, 1.0, 0.6)
    content_weight = 1 - cf_weight

    st.markdown(f"Content Influence: **{round(content_weight,2)}**")

    st.markdown("---")
    st.markdown("### 📊 Dataset Overview")
    st.write(f"🎬 Movies: {len(movies)}")
    st.write(f"⭐ Ratings: {len(ratings)}")

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🎬 Hybrid Movie Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Combining Collaborative Filtering & Content-Based Intelligence</div>", unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    user_id = st.number_input("👤 Enter User ID", 1, 610, 1)

    if user_id in model.user_index_map:
        st.success("✔ Recognized User → Personalized Recommendations")
    else:
        st.warning("⚠ New User → Content-Based Recommendations")

with col2:
    movie_name = st.selectbox("🎥 Select a Movie", movies['title'].values)

st.markdown("---")

# ---------------- BUTTON ----------------
if st.button("🚀 Generate Recommendations"):

    results = model.recommend(user_id, movie_name, cf_weight, content_weight)

    st.markdown("## 🎯 Recommended For You")

    if results.empty:
        st.error("No recommendations found!")
    else:
        cols = st.columns(5)

        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 5]:
                st.markdown(f"""
                <div class="card">
                    <div class="movie-title">{row['title']}</div>
                    <div class="genre">{row['genres']}</div>
                    <div class="score-badge">Score: {round(row['hybrid_score'], 2)}</div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(float(row['hybrid_score']))