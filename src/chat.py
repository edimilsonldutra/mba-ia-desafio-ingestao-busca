import logging

from logging_config import setup_logging
from search import search_prompt
from settings import get_settings

logger = logging.getLogger(__name__)


def main():
    settings = get_settings()

    print("=" * 50)
    print("Chat RAG - Busca Semântica em PDF")
    print("Digite 'sair' para encerrar.")
    print("=" * 50)

    while True:
        pergunta = input("\nFaça sua pergunta: ").strip()

        if not pergunta:
            continue

        if pergunta.lower() == "sair":
            print("Encerrando chat. Até logo!")
            break

        if len(pergunta) > settings.input_max_chars:
            print(f"Pergunta muito longa. Limite: {settings.input_max_chars} caracteres.")
            continue

        try:
            resposta = search_prompt(pergunta)
            if resposta:
                print(f"\nRESPOSTA: {resposta}")
            else:
                print("\nNão foi possível obter uma resposta.")
        except Exception as e:
            logger.exception("Erro ao processar pergunta")
            print("\nErro ao processar pergunta. Verifique os logs para mais detalhes.")


if __name__ == "__main__":
    setup_logging()
    main()