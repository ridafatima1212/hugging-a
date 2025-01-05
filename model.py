from transformers import pipeline

# Load a pre-trained sentiment analysis model
class SentimentModel:
    def __init__(self):
        # Using Hugging Face's pre-trained pipeline
        self.analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", framework="pt")

    def predict(self, text: str) -> str:
        # Get predictions
        results = self.analyzer(text)
        # Extract label and score
        label = results[0]["label"]
        score = results[0]["score"]
        return f"Sentiment: {label}, Confidence: {score:.2f}"

# Instance of the model
model = SentimentModel()
