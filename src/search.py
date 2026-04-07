import logging

from langchain_postgres import PGVector
from tenacity import retry, stop_after_attempt, wait_exponential

from providers import (
    get_embedding_candidates,
    get_embeddings as provider_get_embeddings,
    get_llm as provider_get_llm,
    get_llm_candidates,
    get_provider_order,
)
from settings import get_settings, require_database_url

logger = logging.getLogger(__name__)
FALLBACK_ANSWER = "Não tenho informações necessárias para responder sua pergunta."

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def get_embeddings():
    return provider_get_embeddings()


def get_llm():
    return provider_get_llm()


def _collection_name_for_provider(provider: str | None) -> str:
    settings = get_settings()
    if not provider:
        return settings.collection_name
    return f"{settings.collection_name}_{provider}"


def get_vectorstore(embeddings=None, provider: str | None = None):
    """Create and return the PGVector vectorstore."""
    embeddings = embeddings or get_embeddings()

    return PGVector(
        embeddings=embeddings,
        collection_name=_collection_name_for_provider(provider),
        connection=require_database_url(),
        use_jsonb=True,
    )


def _should_try_next_provider(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = [
        "insufficient_quota",
        "quota",
        "resourceexhausted",
        "rate limit",
        "rate_limit",
        "not found",
        "unsupported",
        "404",
        "429",
    ]
    return any(marker in message for marker in markers)


def _search_with_provider_fallback(question: str) -> list:
    last_exc = None
    seen_stores: set[int] = set()

    # Mantem compatibilidade com mocks existentes de get_vectorstore().
    try:
        provider = get_provider_order()[0]
        vectorstore = get_vectorstore(provider=provider)
        seen_stores.add(id(vectorstore))
        results = vectorstore.similarity_search_with_score(question, k=10)
        if results:
            return results
        logger.info("Provider=%s retornou 0 resultados, tentando próximo provider", provider)
    except Exception as exc:
        last_exc = exc
        if not _should_try_next_provider(exc):
            raise

    for provider, embeddings in get_embedding_candidates():
        try:
            vectorstore = get_vectorstore(embeddings, provider=provider)
            if id(vectorstore) in seen_stores:
                continue
            seen_stores.add(id(vectorstore))
            results = vectorstore.similarity_search_with_score(question, k=10)
            if results:
                logger.info("Busca vetorial executada com provider=%s", provider)
                return results
            logger.info("Provider=%s retornou 0 resultados, tentando próximo provider", provider)
        except Exception as exc:
            last_exc = exc
            if _should_try_next_provider(exc):
                logger.warning("Falha no provider=%s para embeddings, tentando próximo: %s", provider, exc)
                continue
            raise

    if last_exc:
        raise last_exc
    return []


def search(question: str) -> list:
    """Search for the top 10 most relevant chunks for a given question."""
    if not question or not question.strip():
        return []

    return _search_with_provider_fallback(question)


def build_prompt(question: str, results: list) -> str:
    """Build prompt with context from search results."""
    contexto = "\n\n".join([doc.page_content for doc, _score in results])
    return PROMPT_TEMPLATE.format(contexto=contexto, pergunta=question)


def _has_relevant_context(results: list) -> bool:
    settings = get_settings()
    if not results:
        return False

    return any(score <= settings.max_context_distance for _doc, score in results)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=True)
def _invoke_with_retry(llm, prompt: str):
    return llm.invoke(prompt)


def _invoke_with_provider_fallback(prompt: str):
    last_exc = None

    # Mantem compatibilidade com mocks existentes de get_llm().
    try:
        response = _invoke_with_retry(get_llm(), prompt)
        return response
    except Exception as exc:
        last_exc = exc
        if not _should_try_next_provider(exc):
            raise

    for provider, llm in get_llm_candidates():
        try:
            response = _invoke_with_retry(llm, prompt)
            logger.info("LLM executado com provider=%s", provider)
            return response
        except Exception as exc:
            last_exc = exc
            if _should_try_next_provider(exc):
                logger.warning("Falha no provider=%s para LLM, tentando próximo: %s", provider, exc)
                continue
            raise

    if last_exc:
        raise last_exc
    raise RuntimeError("Nenhum provider disponível para execução da LLM")


def _extractive_fallback(results: list) -> str:
    if not results:
        return FALLBACK_ANSWER

    best_doc = results[0][0]
    excerpt = (best_doc.page_content or "").strip().replace("\n", " ")
    if not excerpt:
        return FALLBACK_ANSWER

    excerpt = excerpt[:500]
    return f"Nao foi possivel consultar o modelo externo agora. Trecho mais relevante encontrado: {excerpt}"


def search_prompt(question: str = None) -> str | None:
    """Execute full RAG pipeline: search + LLM call. Returns the answer string."""
    if not question:
        return None

    results = search(question)
    if not _has_relevant_context(results):
        return FALLBACK_ANSWER

    prompt = build_prompt(question, results)
    try:
        response = _invoke_with_provider_fallback(prompt)
        logger.info("Resposta gerada com sucesso para pergunta de tamanho=%s", len(question))
        return response.content
    except Exception as exc:
        logger.warning("Falha em todos os providers de LLM. Usando fallback extrativo: %s", exc)
        return _extractive_fallback(results)