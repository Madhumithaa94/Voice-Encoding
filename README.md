# 🎙️ Voice-Encoding

This project is part of the **StreamLingo VoiceSync** initiative. It focuses on extracting, visualizing, clustering, and tagging speaker embeddings using audio files and pretrained models like **wav2vec2**, **OpenL3**, and **VGGish**.

---

## 📌 Objective

To analyze and compare speaker characteristics using voice embeddings and enable real-time features such as clustering, voice similarity search, and tagging support.

---

## 🧠 Models Used

| Model     | Dimensionality | Source       |
|-----------|----------------|--------------|
| wav2vec2  | 768            | Facebook AI  |
| OpenL3    | 512            | MIT CSAIL    |
| VGGish    | 128            | Google       |

---

## 🚀 Features Implemented

- ✅ Load and compare voice embeddings from `.npy` files  
- ✅ UMAP projection for dimensionality reduction  
- ✅ KMeans and HDBSCAN clustering  
- ✅ Cluster centroid generation  
- ✅ Voice similarity search using cosine similarity  
- ✅ Streamlit-based UI to explore embeddings  
- ✅ Audio tagging and metadata editor  
- ✅ (Experimental) Auto-tagging for:
  - Gender classification
  - Emotion classification (WIP)

---

## 🖼 Sample Outputs

- UMAP Projections  
- Cluster visualizations (KMeans, HDBSCAN)  
- Table mapping each audio to cluster  
- Centroid audio preview (prototype)  
- Similar voice results  
- Editable metadata tags

---

## 🧭 Folder Structure

