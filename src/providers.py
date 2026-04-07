from settings import get_settings


class LocalHashEmbeddings:
    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_text(self, text: str) -> list[float]:
        import hashlib

        vector = [0.0] * self.dim
        if not text:
            return vector

        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vector[idx] += sign

        norm = sum(v * v for v in vector) ** 0.5
        if norm == 0:
            return vector

        return [v / norm for v in vector]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]


def _normalize_provider(name: str) -> str:
    return (name or "").strip().lower()


def get_provider_order() -> list[str]:
    settings = get_settings()
    requested = [_normalize_provider(x) for x in settings.provider_priority.split(",")]
    requested = [x for x in requested if x in {"openai", "google", "local"}]

    # Garantir ordem estável e sem duplicidade.
    ordered: list[str] = []
    for provider in requested + ["google", "openai", "local"]:
        if provider not in ordered:
            ordered.append(provider)

    available: list[str] = []
    if settings.google_api_key:
        available.append("google")
    if settings.openai_api_key:
        available.append("openai")
    if settings.enable_local_fallback:
        available.append("local")

    if not available:
        raise ValueError("Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env")

    return [p for p in ordered if p in available]


def _create_embeddings(provider: str):
    settings = get_settings()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model)

    if provider == "local":
        return LocalHashEmbeddings(dim=settings.local_embedding_dim)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=settings.google_embedding_model)


def _create_llm(provider: str):
    settings = get_settings()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=0,
            timeout=settings.llm_timeout_seconds,
        )

    if provider == "local":
        return None

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.google_chat_model,
        temperature=0,
        timeout=settings.llm_timeout_seconds,
    )


def get_embedding_candidates() -> list[tuple[str, object]]:
    return [(provider, _create_embeddings(provider)) for provider in get_provider_order()]


def get_llm_candidates() -> list[tuple[str, object]]:
    return [
        (provider, llm)
        for provider in get_provider_order()
        for llm in [_create_llm(provider)]
        if llm is not None
    ]


def get_embeddings():
    return get_embedding_candidates()[0][1]


def get_llm():
    return get_llm_candidates()[0][1]
