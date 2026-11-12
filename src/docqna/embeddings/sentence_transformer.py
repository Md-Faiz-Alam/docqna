from sentence_transformers import SentenceTransformer

from docqna.embeddings.base import BaseEmbeddingModel
from docqna.exceptions import EmbeddingError


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    Local embedding model using Sentence Transformers.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise EmbeddingError(f"Could not load embedding model '{model_name}'.") from exc

    @property
    def dimension(self) -> int:
        """Return embedding vector dimensionality."""

        dimension = self.model.get_embedding_dimension()

        if dimension is None:
            raise EmbeddingError("Could not determine embedding dimension.")

        return dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for document texts."""

        if not texts:
            return []

        cleaned_texts = [text.strip() for text in texts]

        if any(not text for text in cleaned_texts):
            raise ValueError("Document texts cannot contain empty strings.")

        try:
            embeddings = self.model.encode(
                cleaned_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError("Failed to generate document embeddings.") from exc

        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a query."""

        text = text.strip()

        if not text:
            raise ValueError("Query text cannot be empty.")

        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError("Failed to generate query embedding.") from exc

        return embedding.tolist()
