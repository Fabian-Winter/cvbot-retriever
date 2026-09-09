# cvbot-retriever

Retrieval and generation for a RAG chatbot: it embeds a question with AWS
Bedrock, looks up the matching document chunks in the ChromaDB filled by
cvbot-embedder and lets a Bedrock LLM generate an answer from them.

## How it works

1. **Embed** – the question is embedded with the same Bedrock model the chunks
   were indexed with.
2. **Retrieve** – the `TOP_K` nearest chunks are read from the ChromaDB
   collection, including their metadata (`source`, `filename`, `chunk_index`).
3. **Generate** – question and chunks are combined into a prompt and sent to
   the Bedrock LLM through the Converse API.

The collection is only read; creating and filling it stays the responsibility
of cvbot-embedder.

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.13 and AWS credentials with access to `bedrock:InvokeModel`
for both the embedding and the LLM model. Credentials are resolved through the
usual boto3 chain (environment variables, profile, or the instance profile of
the host the code runs on).

## Configuration

Configuration is done through environment variables. Every value has a default;
usually only `CHROMA_HOST` needs to be set.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHROMA_HOST` | `localhost` | Hostname of the ChromaDB (Fargate service) |
| `CHROMA_PORT` | `8000` | Port of the ChromaDB |
| `CHROMA_COLLECTION` | `cvbot_documents` | Name of the collection |
| `AWS_REGION` | `eu-central-1` | Region of the Bedrock client |
| `EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock model ID for the question |
| `LLM_MODEL_ID` | `amazon.nova-lite-v1:0` | Bedrock model ID for the answer |
| `TOP_K` | `4` | Number of chunks retrieved per question |
| `LOG_LEVEL` | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARNING` or `ERROR` |

`CHROMA_COLLECTION` and `EMBEDDING_MODEL_ID` must match the values used by
cvbot-embedder, otherwise the query vectors are incompatible with the indexed
ones.

## Usage

```python
from cvbot_retriever.config import Settings
from cvbot_retriever.pipeline import answer_question

result = answer_question(Settings.from_env(), "Which projects has the candidate worked on?")
print(result.answer)
```

For a quick smoke test against a running ChromaDB:

```bash
python -m cvbot_retriever "Which projects has the candidate worked on?" \
  --top-k 6 \
  --log-level DEBUG
```

## Tests

```bash
python -m pytest
```

The tests run without AWS or network access: Bedrock is replaced by a
deterministic embedding model and a Converse double, ChromaDB by a store
double.

## Layout

```
cvbot_retriever/
  config.py        Settings from environment variables
  embeddings.py    Bedrock embedding model for the question
  vector_store.py  Read-only ChromaDB client and collection
  retriever.py     Top-k chunk lookup
  llm.py           Bedrock Converse client for the generation
  pipeline.py      Orchestration of retrieval and generation
  __main__.py      Command line
```

## Known limitations

- One question per call: there is no conversation history and no context
  truncation yet.
- `build_prompt` only concatenates chunks and question; persona, grounding
  rules and prompt injection defenses are still missing.
- Errors from Bedrock or ChromaDB propagate unchanged; there is no retry or
  user-facing error handling yet.

