import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan

def load_embeddings(model):
    files = glob.glob(os.path.join("output_embeddings", model, "*.npy"))
    data, names = [], []
    for f in files:
        data.append(np.load(f).squeeze())
        names.append(os.path.splitext(os.path.basename(f))[0])
    if not data:
        raise FileNotFoundError(f"No embeddings for '{model}'")
    print(f"[{model}] Loaded {len(data)} embeddings (dim={data[0].shape[0]})")
    return np.vstack(data), names

def run_umap(X):
    n = len(X)
    k = max(2, min(n-1, 5))
    emb = umap.UMAP(n_neighbors=k, min_dist=0.1, random_state=42).fit_transform(X)
    return emb, k

def plot_clusters(X2, labels, method, model):
    unique = set(labels)
    n_clust = len(unique) - (1 if -1 in unique else 0)
    sil = silhouette_score(X2, labels) if n_clust >= 2 else float("nan")
    print(f"[{model}] {method} → clusters={n_clust}, silhouette={sil:.3f}")
    plt.figure(figsize=(6,6))
    sc = plt.scatter(X2[:,0], X2[:,1], c=labels, cmap="Spectral", s=50)
    plt.title(f"{model.upper()} – {method} – sil={sil:.3f}")
    plt.legend(*sc.legend_elements(), title="clusters")
    plt.savefig(f"umap_{model}_{method}.png", dpi=150)
    plt.close()

def analyze(model):
    X, names = load_embeddings(model)
    X2, k = run_umap(X)
    pd.DataFrame(X2, columns=["x","y"]).assign(name=names).to_csv(f"umap_coords_{model}.csv", index=False)
    print(f"[{model}] UMAP coords saved (k={k})")

    # HDBSCAN clustering
    h_labels = hdbscan.HDBSCAN(min_cluster_size=2).fit_predict(X2)
    plot_clusters(X2, h_labels, "HDBSCAN", model)

    # K-Means (fixed k=2)
    km_labels = KMeans(n_clusters=2, random_state=42).fit_predict(X2)
    plot_clusters(X2, km_labels, "KMeans_k2", model)

def main():
    for model in ("wav2vec2", "openl3", "vggish"):
        try:
            analyze(model)
        except FileNotFoundError:
            print(f"[WARN] Missing embeddings for '{model}', skipping")
        except Exception as e:
            print(f"[ERROR] in '{model}': {e}")

if __name__ == "__main__":
    main()
