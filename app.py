import os, glob
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap
import hdbscan
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("✅ Voice Embedding UMAP + Clustering")

model = st.selectbox("Choose model", ["wav2vec2", "openl3", "vggish"])
files = glob.glob(os.path.join("output_embeddings", model, "*.npy"))
if not files:
    st.error("No embeddings found.")
    st.stop()

X = np.vstack([np.load(f) for f in files])
st.write("Data shape:", X.shape)

k = max(2, min(len(X) - 1, 5))
X2 = umap.UMAP(n_neighbors=k, min_dist=0.1, random_state=42).fit_transform(X)

fig_u, ax_u = plt.subplots(figsize=(6, 5))
ax_u.scatter(X2[:, 0], X2[:, 1], s=40)
ax_u.set_title("UMAP Projection")
st.pyplot(fig_u, clear_figure=False)

k_labels = KMeans(n_clusters=2, random_state=42).fit_predict(X2)
h_labels = hdbscan.HDBSCAN(min_cluster_size=2).fit_predict(X2)
fig_c, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X2[:, 0], X2[:, 1], c=k_labels, cmap="cool", s=40)
ax1.set_title("KMeans")
ax2.scatter(X2[:, 0], X2[:, 1], c=h_labels, cmap="viridis", s=40)
ax2.set_title("HDBSCAN")
st.pyplot(fig_c, clear_figure=False)

st.subheader("Cluster Assignments")
st.dataframe({
    "File": [os.path.basename(f) for f in files],
    "KMeans Cluster": k_labels,
    "HDBSCAN Cluster": h_labels
})
