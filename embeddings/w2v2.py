import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import torchaudio

class Wav2Vec2Extractor:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(self.device)
        self.target_sr = self.processor.feature_extractor.sampling_rate  # 16000

    def extract(self, filepath):
        audio, sr = sf.read(filepath)
        if sr != self.target_sr:
            audio = torchaudio.functional.resample(torch.from_numpy(audio).float(), sr, self.target_sr).numpy()
            sr = self.target_sr
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        with torch.inference_mode():
            hs = self.model(input_values).last_hidden_state
            emb = hs.mean(dim=1).cpu().numpy()
        return emb

if __name__ == "__main__":
    emb = Wav2Vec2Extractor().extract(r"C:\Users\Abcom\Downloads\harvard_16k_mono.wav")
    print("W2V2 emb shape:", emb.shape)
