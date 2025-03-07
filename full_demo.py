import torch
import clip
import numpy as np
import pandas as pd
import pickle
import faiss
import streamlit as st
import ast
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)


def encode_text(text):
    """Encodes search text using CLIP"""
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([text]).to(device))
    return text_features.cpu().numpy().astype(np.float32)


def encode_image(image_path):
    """Encodes an image using CLIP"""
    image = preprocess(Image.open(image_path).convert("RGB"))
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        image_feature = model.encode_image(image)
    return image_feature.cpu().numpy().astype(np.float32)


def normalize_embeddings(matrix):
    """Normalizes embeddings to unit length for cosine similarity"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / norms


@st.cache_data
def load_data():
    """Loads and processes data, caches embeddings for faster performance"""
    df_1 = pd.read_csv('image_output.csv')
    df_1["Text Embeddings"] = df_1["Text Embeddings"].map(ast.literal_eval).apply(np.array)

    # Load image embeddings
    with open("embeddings.pkl", "rb") as f:
        image_embeddings_matrix = pickle.load(f).astype(np.float32)

    # Convert text embeddings to NumPy array and normalize
    text_embeddings_matrix = np.vstack(df_1["Text Embeddings"].values).astype(np.float32)
    text_embeddings_matrix = normalize_embeddings(text_embeddings_matrix)

    # Normalize image embeddings
    image_embeddings_matrix = normalize_embeddings(image_embeddings_matrix)

    # Build FAISS index for text embeddings
    text_index = faiss.IndexFlatIP(text_embeddings_matrix.shape[1])
    text_index.add(text_embeddings_matrix)

    return df_1, text_index, text_embeddings_matrix, image_embeddings_matrix


df_1, text_index, text_embeddings_matrix, image_embeddings_matrix = load_data()


def search_similar_clothing(query, text_weight=0.5, image_weight=0.5):
    """Search for most similar clothing items using both text and image embeddings"""
    query_embedding = encode_text(query)
    query_embedding = normalize_embeddings(query_embedding)

    # Compute text similarity using FAISS
    _, top_indices_text = text_index.search(query_embedding, len(df_1))
    text_similarities = cosine_similarity(query_embedding, text_embeddings_matrix).flatten()

    # Compute image similarity
    image_similarities = cosine_similarity(query_embedding, image_embeddings_matrix).flatten()

    # Handle mismatched lengths by trimming to the minimum length
    min_length = min(len(text_similarities), len(image_similarities))
    text_similarities = text_similarities[:min_length]
    image_similarities = image_similarities[:min_length]

    # Combine similarities with weighted importance
    combined_similarities = (text_similarities * text_weight) + (image_similarities * image_weight)

    # Get top results
    top_indices = combined_similarities.argsort()[-6:][::-1]
    return df_1.iloc[top_indices]


def search_similar_images(selected_image_path):
    """Finds images similar to a selected image using CLIP image embeddings."""
    query_embedding = encode_image(selected_image_path)
    query_embedding = normalize_embeddings(query_embedding)

    # Compute image similarity
    image_similarities = cosine_similarity(query_embedding, image_embeddings_matrix).flatten()

    # Get top similar images
    top_indices = image_similarities.argsort()[-6:][::-1]
    return df_1.iloc[top_indices]


# Streamlit UI
st.title("LooksGood Visual Search Engine Demo")
st.subheader('Instructions')
st.write('Select your image and text weight to determine how much you want your recommendations to be based on just looks')
st.write('Enter the search query of your choice and indefinitely select your favorite items to hone in on your style')
# Initialize session state if it doesn't exist
if "selected_images" not in st.session_state:
    st.session_state.selected_images = []
if "current_results" not in st.session_state:
    st.session_state.current_results = pd.DataFrame()  # Store latest results

# Handle the query input
search_query = st.text_input("Enter your search query (e.g., 'summer dress for trip in Italy'):")
text_weight = st.slider("Text Weight", 0.0, 1.0, 0.5)
image_weight = 1.0 - text_weight  # Ensure they sum to 1

# When a search query is entered, find similar items
if search_query and st.session_state.current_results.empty:
    st.session_state.current_results = search_similar_clothing(search_query, text_weight, image_weight)

# Display current results
if not st.session_state.current_results.empty:
    new_selection = None  # Track newly selected image
    for _, row in st.session_state.current_results.iterrows():
        col1, col2 = st.columns([4, 1])
        with col1:
            image_path = row["Image Path"]
            if os.path.exists(image_path):
                st.image(image_path, caption=row["Name"], use_column_width=True)
            else:
                st.warning(f"Image not found: {image_path}")
        with col2:
            button_key = f"button_{row['Name']}_{row.name}"
            if st.button(f"Select {row['Name']}", key=button_key):
                st.session_state.selected_images.append(row['Name'])
                new_selection = row["Image Path"]

    # If an image was selected, update current results with similar images
    if new_selection:
        st.session_state.current_results = search_similar_images(new_selection)
        st.rerun()  # Refresh UI with new results

# Display the selection summary (Names Only)
if st.session_state.selected_images:
    st.subheader("Selection Summaries")
    st.write(", ".join(st.session_state.selected_images))  # Show names as a single line

# Button to clear all selections and reset
if st.button("Clear Selections"):
    st.session_state.selected_images = []
    st.session_state.current_results = pd.DataFrame()  # Clear displayed images
    st.rerun()