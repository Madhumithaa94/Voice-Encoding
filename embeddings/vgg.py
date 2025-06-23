import torch
import numpy as np
import torchaudio
from torchaudio.prototype.pipelines import VGGISH  # note uppercase

class VGGishExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = VGGISH.get_input_processor()
        self.model = VGGISH.get_model().to(self.device)
        self.sample_rate = VGGISH.sample_rate

    def extract(self, filepath):
        wav, sr = torchaudio.load(filepath)
        wav = wav.squeeze(0)  # mono
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        inputs = self.processor(wav)  # shape (N, ...)
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            embs = self.model(inputs)  # shape (N, 128)
        return embs.mean(dim=0, keepdim=True).cpu().numpy()
