# embeddings/be.py
import os
import numpy as np
from w2v2 import Wav2Vec2Extractor
from open import OpenL3Extractor
from vgg import VGGishExtractor  # updated import

AUDIO_FILES = [
    r"C:\Users\Abcom\Downloads\harvard_16k_mono.wav",
    r"C:\Users\Abcom\Downloads\OSR_us_000_0012_8k.wav",
    r"C:\Users\Abcom\Downloads\OSR_us_000_0011_8k.wav",
    r"C:\Users\Abcom\Downloads\OSR_us_000_0010_8k.wav",
]

extractors = {
    "wav2vec2": Wav2Vec2Extractor(),
    "openl3": OpenL3Extractor(),
    "vggish": VGGishExtractor(),  # instantiate from vgg.py
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output_embeddings")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for name, ext in extractors.items():
    model_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    for f in AUDIO_FILES:
        if not os.path.exists(f):
            print(f"[SKIP] Missing file: {f}")
            continue
        print(f"[{name}] extracting from {os.path.basename(f)}...")
        emb = ext.extract(f)
        out = os.path.join(model_dir, os.path.splitext(os.path.basename(f))[0] + ".npy")
        np.save(out, emb)
        print(f"→ saved shape {emb.shape} to {out}")
