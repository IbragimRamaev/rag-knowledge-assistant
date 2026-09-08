from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Turns text into vector embeddings. Swap implementations without touching callers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
