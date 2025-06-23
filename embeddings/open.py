# embeddings/open.py
import openl3
import soundfile as sf
import numpy as np

class OpenL3Extractor:
    def __init__(self,
                 content_type="music",  # <-- or "env"
                 embedding_size=512,
                 input_repr="mel256"):
        self.content_type = content_type
        self.embedding_size = embedding_size
        self.input_repr = input_repr

    def extract(self, filepath):
        audio, sr = sf.read(filepath)
        emb_frames, _ = openl3.get_audio_embedding(
            audio, sr,
            input_repr=self.input_repr,
            content_type=self.content_type,
            embedding_size=self.embedding_size,
            hop_size=0.1,  # or adjust as needed
            verbose=0
        )
        return emb_frames.mean(axis=0, keepdims=True)
