import os
import streamlit as st
import pandas as pd
import plotly.express as px
import spacy
from umap import UMAP
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from collections import Counter

# --- Page Config ---
st.set_page_config(page_title="LinguisticAtlas", layout="wide")

# --- Resource Loading (Suggestion 1: Robust Deployment Loading) ---
@st.cache_resource
def load_models():
    model_name = "en_core_web_sm"
    try:
        # Check if the model is already installed
        nlp = spacy.load(model_name)
    except OSError:
        # If not found, download it automatically on the server
        st.write(f"Downloading {model_name} for the first time...")
        os.system(f"python -m spacy download {model_name}")
        nlp = spacy.load(model_name)
    
    # Load embedding model for semantic analysis
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, embed_model

@st.cache_data
def get_data():
    # Load a linguistic dataset from Hugging Face
    dataset = load_dataset("tweet_eval", "sentiment", split='train[:2000]')
    df = pd.DataFrame(dataset)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    df['sentiment'] = df['label'].map(label_map)
    return df

# Initialize Models and Data
nlp, embed_model = load_models()
df = get_data()

# --- Sidebar Navigation (Suggestion 2: Aligned Navigation Logic) ---
st.sidebar.title("🔍 LinguisticAtlas") 
st.sidebar.markdown("**An Interactive Instrument for Systematic Linguistic Data Analysis.**")
st.sidebar.divider()

# These strings MUST match the if/elif conditions below exactly
page = st.sidebar.radio("Project Modules:", ["Semantic Space (UMAP)", "Morphosyntactic Analysis", "Data Explorer"])

# --- Page 1: UMAP Semantic Space ---
if page == "Semantic Space (UMAP)":
    st.header("🌐 Global Semantic Space")
    st.info("This visualization projects 384-dimensional embeddings into a 2D manifold to reveal latent semantic clusters.")
    
    with st.spinner("Calculating embeddings and UMAP coordinates..."):
        # Vectorize the text
        embeddings = embed_model.encode(df['text'].tolist())
        # Dimensionality Reduction
        reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        umap_results = reducer.fit_transform(embeddings)
        df['x'], df['y'] = umap_results[:, 0], umap_results[:, 1]

    fig = px.scatter(
        df, x='x', y='y', color='sentiment',
        hover_data={'text': True, 'x': False, 'y': False},
        color_discrete_map={"Negative": "#EF553B", "Neutral": "#636EFA", "Positive": "#00CC96"},
        height=600,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Page 2: Morphosyntactic Analysis ---
elif page == "Morphosyntactic Analysis":
    st.header("🔤 Morphosyntactic Analysis")
    st.markdown("Analyze the distribution of **lemmas** and **Part-of-Speech (POS)** tags across the corpus.")
    
    target_sentiment = st.selectbox("Select Sentiment Group:", ["All", "Positive", "Negative", "Neutral"])
    filtered_df = df if target_sentiment == "All" else df[df['sentiment'] == target_sentiment]

    all_lemmas = []
    pos_counts = []
    
    with st.spinner("Executing NLP Pipeline (Tokenization, Lemmatization, POS Tagging)..."):
        # nlp.pipe is more efficient for large batches
        for doc in nlp.pipe(filtered_df['text'].astype(str)):
            for token in doc:
                # Linguistic filtering: Remove noise but keep core lexical units
                if not token.is_stop and not token.is_punct and not token.like_url:
                    all_lemmas.append(token.lemma_.lower())
                    pos_counts.append(token.pos_)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Lemmas")
        lemma_df = pd.DataFrame(Counter(all_lemmas).most_common(15), columns=['Word', 'Count'])
        st.bar_chart(data=lemma_df, x='Word', y='Count')

    with col2:
        st.subheader("POS Distribution")
        pos_df = pd.DataFrame(Counter(pos_counts).most_common(10), columns=['POS', 'Count'])
        fig_pos = px.pie(pos_df, values='Count', names='POS', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pos)

# --- Page 3: Data Explorer ---
elif page == "Data Explorer":
    st.header("📄 Dataset View")
    st.markdown("Review the raw textual data and associated metadata labels.")
    st.dataframe(df[['text', 'sentiment']], use_container_width=True)