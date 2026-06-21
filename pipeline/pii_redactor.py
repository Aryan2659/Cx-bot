"""
PII Redaction using Microsoft Presidio.
Falls back to regex-based redaction if Presidio is unavailable.
"""
import re
import logging

logger = logging.getLogger(__name__)

# Try to load Presidio; graceful fallback if not installed
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    _analyzer = AnalyzerEngine()
    _anonymizer = AnonymizerEngine()
    PRESIDIO_AVAILABLE = True
    logger.info("Presidio loaded successfully.")
except Exception as e:
    PRESIDIO_AVAILABLE = False
    logger.warning(f"Presidio not available ({e}). Using regex fallback.")

# Regex patterns for fallback
_PATTERNS = [
    (re.compile(r"\b[\w.-]+@[\w.-]+\.\w{2,}\b"), "<EMAIL>"),
    (re.compile(r"\b(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"), "<PHONE>"),
    (re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b"), "<CREDIT_CARD>"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
]

PRESIDIO_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IP_ADDRESS",
    "LOCATION",
]


def redact(text: str) -> str:
    """Return text with PII replaced by placeholders."""
    if not text:
        return text

    if PRESIDIO_AVAILABLE:
        try:
            results = _analyzer.analyze(
                text=text, entities=PRESIDIO_ENTITIES, language="en"
            )
            anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
            return anonymized.text
        except Exception as e:
            logger.error(f"Presidio redaction failed: {e}. Using regex fallback.")

    # Regex fallback
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
