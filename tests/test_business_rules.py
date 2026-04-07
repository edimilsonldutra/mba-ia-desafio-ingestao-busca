"""
Testes de regras de negócio para o sistema de Ingestão e Busca Semântica.
Valida o comportamento esperado do sistema conforme os requisitos.
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from search import build_prompt, PROMPT_TEMPLATE
from ingest import PDF_PATH


class TestChunkingRules(unittest.TestCase):
    """Testa as regras de chunking do PDF."""

    def test_splitter_config_chunk_size_1000(self):
        """Chunks devem ter no máximo 1000 caracteres."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        text = "A" * 3000
        docs = [Document(page_content=text)]
        chunks = splitter.split_documents(docs)

        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 1000)

    def test_splitter_config_overlap_150(self):
        """Chunks devem ter overlap de 150 caracteres."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        text = "A" * 2000
        docs = [Document(page_content=text)]
        chunks = splitter.split_documents(docs)

        self.assertGreater(len(chunks), 1)
        # Verify overlap exists between consecutive chunks
        overlap = set(chunks[0].page_content[-150:]) & set(chunks[1].page_content[:150])
        self.assertTrue(len(overlap) > 0)

    def test_splitter_generates_multiple_chunks(self):
        """Um documento grande deve gerar múltiplos chunks."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        text = "Texto de exemplo. " * 500  # ~9500 chars
        docs = [Document(page_content=text)]
        chunks = splitter.split_documents(docs)

        self.assertGreater(len(chunks), 1)


class TestPromptBuildRules(unittest.TestCase):
    """Testa a construção do prompt com contexto e regras."""

    def test_prompt_contains_context(self):
        """O prompt deve incluir o contexto dos documentos encontrados."""
        results = [
            (Document(page_content="Faturamento foi de 10 milhões."), 0.1),
            (Document(page_content="Empresa foi fundada em 2020."), 0.2),
        ]
        prompt = build_prompt("Qual o faturamento?", results)

        self.assertIn("Faturamento foi de 10 milhões.", prompt)
        self.assertIn("Empresa foi fundada em 2020.", prompt)

    def test_prompt_contains_question(self):
        """O prompt deve incluir a pergunta do usuário."""
        results = [(Document(page_content="Texto qualquer."), 0.1)]
        prompt = build_prompt("Qual o faturamento?", results)

        self.assertIn("Qual o faturamento?", prompt)

    def test_prompt_contains_rules(self):
        """O prompt deve conter as regras para resposta restrita ao contexto."""
        results = [(Document(page_content="Texto."), 0.1)]
        prompt = build_prompt("Pergunta?", results)

        self.assertIn("Responda somente com base no CONTEXTO", prompt)
        self.assertIn("Não tenho informações necessárias para responder sua pergunta", prompt)
        self.assertIn("Nunca invente ou use conhecimento externo", prompt)

    def test_prompt_contains_out_of_context_examples(self):
        """O prompt deve conter exemplos do que é uma pergunta fora do contexto."""
        results = [(Document(page_content="Texto."), 0.1)]
        prompt = build_prompt("Pergunta?", results)

        self.assertIn("Qual é a capital da França?", prompt)
        self.assertIn("Quantos clientes temos em 2024?", prompt)

    def test_prompt_with_empty_results(self):
        """Prompt com resultados vazios deve ter contexto vazio."""
        prompt = build_prompt("Pergunta?", [])
        self.assertIn("CONTEXTO:", prompt)
        self.assertIn("Pergunta?", prompt)


class TestSearchPromptRules(unittest.TestCase):
    """Testa o pipeline de busca e resposta."""

    def test_search_prompt_returns_none_for_empty_question(self):
        """Sem pergunta, search_prompt deve retornar None."""
        from search import search_prompt

        result = search_prompt(None)
        self.assertIsNone(result)

        result = search_prompt("")
        self.assertIsNone(result)

    @patch("search.get_llm")
    @patch("search.get_vectorstore")
    def test_search_uses_k10(self, mock_vs, mock_llm):
        """Busca deve ser feita com k=10 resultados."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        mock_vs.return_value = mock_store

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="Resposta")
        mock_llm.return_value = mock_llm_instance

        from search import search

        search("pergunta teste")

        mock_store.similarity_search_with_score.assert_called_once_with(
            "pergunta teste", k=10
        )

    @patch("search.get_llm")
    @patch("search.get_vectorstore")
    def test_search_prompt_returns_llm_response(self, mock_vs, mock_llm):
        """search_prompt deve retornar o conteúdo da resposta da LLM."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (Document(page_content="Faturamento: 10 milhões."), 0.1),
        ]
        mock_vs.return_value = mock_store

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(
            content="O faturamento foi de 10 milhões."
        )
        mock_llm.return_value = mock_llm_instance

        from search import search_prompt

        result = search_prompt("Qual o faturamento?")
        self.assertEqual(result, "O faturamento foi de 10 milhões.")

    @patch("search.get_llm")
    @patch("search.get_vectorstore")
    def test_prompt_sent_to_llm_has_context_and_question(self, mock_vs, mock_llm):
        """O prompt enviado à LLM deve conter contexto e pergunta."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (Document(page_content="Receita anual: 5 milhões."), 0.05),
        ]
        mock_vs.return_value = mock_store

        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = MagicMock(content="Resposta")
        mock_llm.return_value = mock_llm_instance

        from search import search_prompt

        search_prompt("Qual a receita?")

        call_args = mock_llm_instance.invoke.call_args[0][0]
        self.assertIn("Receita anual: 5 milhões.", call_args)
        self.assertIn("Qual a receita?", call_args)

    @patch("search.get_llm")
    @patch("search.get_vectorstore")
    @patch.dict(os.environ, {"MAX_CONTEXT_DISTANCE": "0.5"}, clear=False)
    def test_search_prompt_returns_fallback_for_low_relevance(self, mock_vs, mock_llm):
        """Com score ruim, deve retornar fallback sem chamar LLM."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (Document(page_content="Texto irrelevante"), 0.99),
        ]
        mock_vs.return_value = mock_store

        from search import search_prompt, FALLBACK_ANSWER

        result = search_prompt("Pergunta fora de contexto?")
        self.assertEqual(result, FALLBACK_ANSWER)
        mock_llm.assert_not_called()


class TestIngestConfig(unittest.TestCase):
    """Testa configurações do módulo de ingestão."""

    def test_pdf_path_defaults(self):
        """PDF_PATH deve ter um valor padrão se não definido no .env."""
        self.assertIsNotNone(PDF_PATH)

    @patch("ingest.get_embeddings")
    @patch("ingest.PyPDFLoader")
    @patch("ingest.PGVector")
    @patch.dict(os.environ, {"DATABASE_URL": "postgresql+psycopg://postgres:postgres@localhost:5432/rag"}, clear=False)
    def test_ingest_splits_and_stores_chunks(self, mock_pgvector, mock_loader, mock_emb):
        """Ingestão deve carregar PDF, dividir em chunks e armazenar no banco."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            Document(page_content="Texto " * 500, metadata={"page": 0}),
        ]
        mock_loader.return_value = mock_loader_instance

        mock_emb.return_value = MagicMock()

        mock_store = MagicMock()
        mock_pgvector.return_value = mock_store

        from ingest import ingest_pdf

        ingest_pdf()

        mock_loader.assert_called_once()
        mock_store.add_documents.assert_called_once()

        stored_chunks = mock_store.add_documents.call_args[0][0]
        self.assertGreater(len(stored_chunks), 1)
        for chunk in stored_chunks:
            self.assertLessEqual(len(chunk.page_content), 1000)


class TestEmbeddingsConfig(unittest.TestCase):
    """Testa a seleção de embeddings conforme variáveis de ambiente."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "", "GOOGLE_API_KEY": "", "ENABLE_LOCAL_FALLBACK": "false"}, clear=False)
    def test_no_api_key_raises_error(self):
        """Sem API key configurada, deve lançar ValueError."""
        from ingest import get_embeddings

        with self.assertRaises(ValueError):
            get_embeddings()


class TestSettingsValidation(unittest.TestCase):
    """Testa validação de configurações obrigatórias."""

    @patch.dict(os.environ, {"DATABASE_URL": ""}, clear=False)
    def test_missing_database_url_raises_error(self):
        """Sem DATABASE_URL, deve falhar cedo com erro claro."""
        from settings import require_database_url

        with self.assertRaises(ValueError):
            require_database_url()


if __name__ == "__main__":
    unittest.main()
