echo ^> README.md && (
echo # ^🎙️ Voice-Encoding
echo. 
echo This project is part of the StreamLingo VoiceSync initiative and focuses on **extracting voice embeddings** from audio files using pretrained models. The goal is to visualize and analyze voice characteristics through clustering and dimensionality reduction techniques.
echo.
echo ## ^📌 Project Objective
echo.
echo To generate **speaker embeddings** using pretrained models (like wav2vec2, VGGish, and OpenL3), apply **UMAP** for dimensionality reduction, and visualize clustering results with **KMeans** and **HDBSCAN**. This supports real-time voice preservation and personalization tasks in voice translation systems.
echo.
echo ## ^🧠 Models Used
echo.
echo ^| Model     ^| Dimensionality ^| Source      ^|
echo ^|-----------^|----------------^|-------------^|
echo ^| **wav2vec2** ^| 768              ^| Facebook AI ^|
echo ^| **OpenL3**   ^| 512              ^| MIT CSAIL    ^|
echo ^| **VGGish**   ^| 128              ^| Google       ^|
echo.
echo These models convert audio into fixed-length feature vectors (embeddings), capturing **voice identity, tone, and speaker characteristics**.
echo.
echo ## ^🛠 Features Implemented
echo.
echo - ✅ Load embeddings for selected model (wav2vec2, openl3, vggish)
echo - ✅ Apply **UMAP** for 2D projection
echo - ✅ Perform clustering using:
echo   - **KMeans (k=2)**
echo   - **HDBSCAN (density-based)**
echo - ✅ Visualize UMAP projections with cluster labels
echo - ✅ File-to-cluster mapping table
echo - ✅ Streamlit interface for interaction
echo.
echo ## ^📂 Folder Structure
echo.
echo \`\`\`
echo PROJECT003/
echo ^│
echo ^├── app.py                   # Streamlit frontend
echo ^├── umap_analysis.py         # UMAP + clustering script
echo ^├── output_embeddings/       # .npy embedding files
echo ^├── umap_coords_*.csv        # Optional: saved UMAP coordinates
echo ^├── *.png                    # Saved visualizations
echo ^└── requirements.txt         # Python dependencies (optional)
echo \`\`\`
echo.
echo ## ^🚀 How to Run
echo.
echo ### Install Requirements
echo \`\`\`bash
echo pip install -r requirements.txt
echo \`\`\`
echo.
echo ### Run Streamlit App
echo \`\`\`bash
echo streamlit run app.py
echo \`\`\`
echo.
echo ## ^📊 Sample Output
echo.
echo - UMAP Projections of embeddings
echo - KMeans & HDBSCAN clustering visualized
echo - Table mapping each audio file to clusters
echo.
echo ## ^🤖 Contributions
echo.
echo **Developed by:**  
echo `Madhumithaa94`  
echo as part of an academic and research internship for **Speaker Encoding** in the **StreamLingo VoiceSync** project.
echo.
echo ## ^📌 Notes
echo.
echo - Real-time support not required for this phase.
echo - Himabindu and I worked in divided modules. This work covers the **embedding, UMAP, and clustering visualizations.**
echo.
echo ## ^📎 License
echo.
echo This project is for academic and research purposes only.
) >> README.md
