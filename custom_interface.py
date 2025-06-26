from speechbrain.pretrained import EncoderClassifier

class CustomClassifier(EncoderClassifier):
    def classify_file(self, path):
        out_prob, score, index, text_lab = super().classify_file(path)
        return {"predicted_label": text_lab}
