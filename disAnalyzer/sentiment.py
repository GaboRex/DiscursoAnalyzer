from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentClassifier:
    def __init__(self):
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def classify_sentiment(self, text: str) -> str:
        inputs = self.tokenizer.encode_plus(text, return_tensors='pt', truncation=True, padding=True)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        output = self.model(**inputs)
        _, predicted = torch.max(output.logits, 1)
        sentiment = ["Muy negativo", "Negativo", "Neutral", "Positivo", "Muy positivo"][predicted.item()]
        return sentiment
