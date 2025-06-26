import os, glob, sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.cluster import KMeans
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("🎙️ Voice Embedding UMAP + Clustering + Search")

# Select model
model = st.selectbox("Choose embedding model", ["wav2vec2", "openl3", "vggish"])
embed_dir = os.path.join("output_embeddings", model)
files = sorted(glob.glob(os.path.join(embed_dir, "*.npy")))

if not files:
    st.error(f"No embeddings found in {embed_dir}")
    st.stop()

# Load data
X = np.vstack([np.load(f) for f in files])
names = [os.path.basename(f) for f in files]
st.write("✅ Loaded embeddings shape:", X.shape)

# UMAP projection
k = max(2, min(len(X) - 1, 5))
X2 = umap.UMAP(n_neighbors=k, min_dist=0.1, random_state=42).fit_transform(X)
fig_u, ax_u = plt.subplots(figsize=(6, 5))
ax_u.scatter(X2[:, 0], X2[:, 1], s=40)
ax_u.set_title("UMAP Projection")
st.pyplot(fig_u, clear_figure=False)

# Clustering
k_labels = KMeans(n_clusters=2, random_state=42).fit_predict(X2)
h_labels = hdbscan.HDBSCAN(min_cluster_size=2).fit_predict(X2)

# Plot clusters
fig_c, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X2[:, 0], X2[:, 1], c=k_labels, cmap="cool", s=40)
ax1.set_title("KMeans Clustering")
ax2.scatter(X2[:, 0], X2[:, 1], c=h_labels, cmap="viridis", s=40)
ax2.set_title("HDBSCAN Clustering")
st.pyplot(fig_c, clear_figure=False)

# Cluster table
st.subheader("📋 File-to-Cluster Mapping")
st.dataframe({
    "File": names,
    "KMeans Cluster": k_labels,
    "HDBSCAN Cluster": h_labels
})

# ➕ Centroid embeddings display
st.subheader("📍 KMeans Cluster Centroid Embeddings")
for cid in np.unique(k_labels):
    indices = np.where(k_labels == cid)[0]
    centroid = np.mean(X[indices], axis=0)
    st.write(f"Centroid for Cluster {cid}:")
    st.code(centroid[:10])  # first 10 values only

# 🔎 Similarity search
st.subheader("🔍 Voice Similarity Search")
uploaded = st.file_uploader("Upload a .wav file to find similar voices", type=["wav"])

if uploaded:
    # Pick correct model
    sys.path.append(os.path.join(os.path.dirname(__file__), "embeddings"))
    if model == "wav2vec2":
        from w2v2 import Wav2Vec2Extractor as Extractor
    elif model == "openl3":
        from open import OpenL3Extractor as Extractor
    elif model == "vggish":
        from vgg import VGGishExtractor as Extractor
    else:
        st.error("Unsupported model.")
        st.stop()

    # Save uploaded file and extract
    with open("temp_input.wav", "wb") as f:
        f.write(uploaded.read())
    extractor = Extractor()
    emb = extractor.extract("temp_input.wav").reshape(1, -1)
    os.remove("temp_input.wav")

    # Compute similarity
    sims = cosine_similarity(emb, X)[0]
    top_idx = np.argsort(sims)[::-1][:3]

    st.success("Top 3 Closest Matches:")
    for i in top_idx:
        st.write(f"{names[i]} — similarity score: {sims[i]:.3f}")
