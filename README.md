# Desafio MBA Engenharia de Software com IA - Full Cycle

## Ingestão e Busca Semântica com LangChain e Postgres

Sistema RAG (Retrieval-Augmented Generation) que permite ingerir documentos PDF e realizar buscas semânticas via CLI.

---

## Pré-requisitos

- Python 3.11+
- Docker e Docker Compose
- Chave de API: OpenAI (`OPENAI_API_KEY`) **ou** Google Gemini (`GOOGLE_API_KEY`)

---

## Como executar

### 1. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e preencha **pelo menos uma** API key:

```
OPENAI_API_KEY=sk-...
# ou
GOOGLE_API_KEY=AIza...

# Opcional: prioridade de provider (recomendado para free tier)
LLM_PROVIDER_PRIORITY=google,openai

# Opcional: modelo de embedding do Google (compatível testado)
GOOGLE_EMBEDDING_MODEL=models/gemini-embedding-001
```

Observacao para plano gratuito:
- OpenAI pode retornar erro de quota (`insufficient_quota`) sem billing ativo.
- Google Gemini no free tier tambem tem limites de cota por minuto/dia.
- O projeto tenta fallback automatico para o proximo provider quando encontrar erros de quota/modelo nao suportado.
- Se todos os providers externos falharem, o projeto pode usar fallback local (embeddings locais e resposta extrativa) com:
	`ENABLE_LOCAL_FALLBACK=true`

### 2. Subir o banco de dados

```bash
docker compose up -d
```

Aguarde o PostgreSQL iniciar e a extensão pgVector ser criada.

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Ingestão do PDF

Coloque o PDF na raiz do projeto como `document.pdf` (ou configure `PDF_PATH` no `.env`) e execute:

```bash
cd src
python ingest.py
```

### 5. Chat via CLI

```bash
cd src
python chat.py
```

Exemplo de uso:

```
Faça sua pergunta: Qual o faturamento da empresa?
RESPOSTA: O faturamento foi de 10 milhões de reais.

Faça sua pergunta: Qual é a capital da França?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

Faça sua pergunta: sair
Encerrando chat. Até logo!
```

---

## Testes

```bash
python -m pytest tests/ -v
```

---

## Estrutura do projeto

```
├── docker-compose.yml        # PostgreSQL + pgVector
├── requirements.txt          # Dependências Python
├── .env.example              # Template de variáveis de ambiente
├── document.pdf              # PDF para ingestão
├── src/
│   ├── ingest.py             # Script de ingestão do PDF
│   ├── search.py             # Busca semântica + chamada LLM
│   ├── chat.py               # CLI para interação com usuário
│   └── prompt.md             # Especificação do projeto
├── tests/
│   └── test_business_rules.py # Testes de regras de negócio
└── README.md
```