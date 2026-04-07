import os
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

from logging_config import setup_logging
from providers import (
    get_embedding_candidates,
    get_embeddings as provider_get_embeddings,
    get_provider_order,
)
from settings import get_settings, require_database_url

logger = logging.getLogger(__name__)
PDF_PATH = get_settings().pdf_path


def get_embeddings():
    return provider_get_embeddings()


def _collection_name_for_provider(provider: str | None) -> str:
    settings = get_settings()
    if not provider:
        return settings.collection_name
    return f"{settings.collection_name}_{provider}"


def _validate_pdf_path(pdf_path: str) -> None:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

    if not os.path.isfile(pdf_path):
        raise ValueError(f"PDF_PATH deve apontar para um arquivo: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"PDF_PATH deve ser um arquivo .pdf: {pdf_path}")


def ingest_pdf():
    settings = get_settings()
    _validate_pdf_path(settings.pdf_path)

    logger.info("Carregando PDF: %s", settings.pdf_path)
    loader = PyPDFLoader(settings.pdf_path)
    documents = loader.load()
    logger.info("Páginas carregadas: %s", len(documents))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Chunks gerados: %s", len(chunks))

    last_exc = None

    # Mantem compatibilidade com mocks existentes de get_embeddings().
    try:
        primary_provider = get_provider_order()[0]
        embeddings = get_embeddings()
        vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=_collection_name_for_provider(primary_provider),
            connection=require_database_url(),
            use_jsonb=True,
        )
        vectorstore.add_documents(chunks)
        logger.info("Ingestão concluída com provider primário! %s chunks armazenados no banco de dados.", len(chunks))
        return
    except Exception as exc:
        last_exc = exc
        logger.warning("Falha na ingestão com provider primário: %s", exc)

    for provider, embeddings in get_embedding_candidates():
        try:
            vectorstore = PGVector(
                embeddings=embeddings,
                collection_name=_collection_name_for_provider(provider),
                connection=require_database_url(),
                use_jsonb=True,
            )
            vectorstore.add_documents(chunks)
            logger.info(
                "Ingestão concluída com provider=%s! %s chunks armazenados no banco de dados.",
                provider,
                len(chunks),
            )
            return
        except Exception as exc:
            last_exc = exc
            logger.warning("Falha na ingestão com provider=%s: %s", provider, exc)

    if last_exc:
        raise last_exc


if __name__ == "__main__":
    setup_logging()
    ingest_pdf()