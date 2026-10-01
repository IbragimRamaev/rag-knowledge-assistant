from pathlib import Path

from app.core.models import Document

SUPPORTED_EXTENSIONS = {".txt", ".md"}


def load_documents(documents_dir: Path | str) -> list[Document]:
    """Read every .txt/.md file in a directory (non-recursive) into a Document."""
    documents_dir = Path(documents_dir)
    documents = []
    for path in sorted(documents_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(
                Document(source_document=path.name, text=path.read_text(encoding="utf-8"))
            )
    return documents
