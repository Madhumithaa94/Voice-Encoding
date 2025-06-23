import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "embeddings"))

from w2v2 import Wav2Vec2Extractor
from open import OpenL3Extractor
from vgg import VGGishExtractor

FILE = r"C:\Users\Abcom\Downloads\harvard_16k_mono.wav"

for cls, name in [
    (Wav2Vec2Extractor, "W2V2"),
    (OpenL3Extractor, "OpenL3"),
    (VGGishExtractor, "VGG")
]:
    ext = cls()
    emb = ext.extract(FILE)
    print(f"{name} embedding shape: {emb.shape}")
