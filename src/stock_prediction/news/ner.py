"""Named Entity Recognition and stock entity linking."""

import spacy

from stock_prediction.config import get_setting
from stock_prediction.utils.constants import COMPANY_ALIASES, TICKER_TO_NAME
from stock_prediction.utils.logging import get_logger

logger = get_logger("news.ner")


class StockEntityLinker:
    """Links named entities in text to NIFTY 50 stock tickers."""

    def __init__(self):
        self.spacy_model = get_setting("ner", "spacy_model", default="en_core_web_sm")
        self._nlp = None

    def _load_model(self):
        if self._nlp is not None:
            return
        logger.info(f"Loading spaCy model: {self.spacy_model}")
        self._nlp = spacy.load(self.spacy_model)

    def extract_entities(self, text: str) -> list[dict]:
        """Extract named entities from text."""
        self._load_model()
        doc = self._nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PERSON", "GPE", "PRODUCT"):
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
        return entities

    def link_to_stocks(self, text: str) -> list[str]:
        """Extract stock tickers mentioned in text via NER + alias matching.

        Returns list of matched ticker symbols (e.g., ['RELIANCE.NS', 'TCS.NS']).
        """
        matched_tickers: set[str] = set()

        # Method 1: Direct alias matching on full text (case-insensitive)
        text_lower = text.lower()
        for alias, ticker in COMPANY_ALIASES.items():
            if alias in text_lower:
                matched_tickers.add(ticker)

        # Method 2: NER entity matching
        entities = self.extract_entities(text)
        for entity in entities:
            entity_lower = entity["text"].lower().strip()
            if entity_lower in COMPANY_ALIASES:
                matched_tickers.add(COMPANY_ALIASES[entity_lower])

        # Method 3: Check for ticker symbols directly (e.g., "TCS", "INFY")
        for ticker, name in TICKER_TO_NAME.items():
            symbol_short = ticker.replace(".NS", "")
            if symbol_short in text:
                matched_tickers.add(ticker)

        return list(matched_tickers)

    def link_articles_to_stocks(
        self, articles: list[dict]
    ) -> dict[str, list[dict]]:
        """Map a list of articles to their mentioned stocks.

        Args:
            articles: list of dicts with at least 'title' and optionally 'snippet' keys.

        Returns:
            Dict mapping ticker -> list of matching articles.
        """
        stock_articles: dict[str, list[dict]] = {}

        for article in articles:
            text = article.get("title", "") + " " + article.get("snippet", "")
            tickers = self.link_to_stocks(text)
            for ticker in tickers:
                stock_articles.setdefault(ticker, []).append(article)

        return stock_articles
