"""FinBERT-based financial sentiment analysis."""

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from stock_prediction.config import get_setting
from stock_prediction.utils.logging import get_logger

logger = get_logger("news.sentiment")


@dataclass
class SentimentResult:
    """Sentiment analysis result for a single text."""

    label: str  # positive, negative, neutral
    score: float  # confidence score (0-1)
    positive_score: float
    negative_score: float
    neutral_score: float


class FinancialSentimentAnalyzer:
    """Sentiment analyzer using ProsusAI/finbert."""

    def __init__(self):
        self.model_name = get_setting("sentiment", "model_name", default="ProsusAI/finbert")
        self.batch_size = get_setting("sentiment", "batch_size", default=32)
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return
        logger.info(f"Loading sentiment model: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        # Labels: positive, negative, neutral
        self._labels = ["positive", "negative", "neutral"]

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text."""
        results = self.analyze_batch([text])
        return results[0]

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts."""
        self._load_model()
        results: list[SentimentResult] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Truncate long texts
            batch = [t[:512] for t in batch]

            inputs = self._tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            for j in range(len(batch)):
                scores = probs[j].tolist()
                label_idx = scores.index(max(scores))
                results.append(
                    SentimentResult(
                        label=self._labels[label_idx],
                        score=scores[label_idx],
                        positive_score=scores[0],
                        negative_score=scores[1],
                        neutral_score=scores[2],
                    )
                )

        return results

    def get_sentiment_score(self, text: str) -> float:
        """Get a single sentiment score: positive (>0) to negative (<0)."""
        result = self.analyze(text)
        return result.positive_score - result.negative_score
