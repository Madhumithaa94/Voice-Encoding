# 🎙️ Voice-Encoding

This project is part of the **StreamLingo VoiceSync** initiative and focuses on extracting speaker-specific voice embeddings from audio files using pretrained models. It enables clustering and visualization of voice characteristics to support real-time voice preservation and translation systems.

---

## 📌 Project Objective

To extract speaker embeddings using pretrained models, apply dimensionality reduction (UMAP), and visualize the results through clustering algorithms. This helps identify speaker-specific patterns in audio data.

---

## 🧠 Models Used

| Model     | Embedding Dim | Source       |
|-----------|----------------|--------------|
| wav2vec2  | 768            | Facebook AI  |
| OpenL3    | 512            | MIT CSAIL    |
| VGGish    | 128            | Google       |

These models convert raw audio into fixed-length embeddings that capture tone, timbre, and speaker identity.

---

## 🛠️ Features Implemented

- ✅ Load embeddings from `.npy` files for selected model
- ✅ Apply UMAP for 2D projection
- ✅ Perform clustering using:
  - KMeans (k=2)
  - HDBSCAN (density-based)
- ✅ Visualize clusters using matplotlib (scatter plots)
- ✅ Interactive selection of embedding model via Streamlit
- ✅ Display mapping of audio file to cluster assignment

---

## 📂 Folder Structure

