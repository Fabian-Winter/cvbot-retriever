# cvbot-retriever

Retrieval and generation for a RAG chatbot: it embeds a question with AWS
Bedrock, looks up the matching document chunks in the ChromaDB filled by
cvbot-embedder and lets a Bedrock LLM generate an answer from them - across
several turns of a conversation.

## How it works

1. **Embed** – the question is embedded with the same Bedrock model the chunks
   were indexed with.
2. **Retrieve** – the `TOP_K` nearest chunks are read from the ChromaDB
   collection, including their metadata (`source`, `filename`, `chunk_index`).
3. **Build the context** – chunks and question become the current user message;
   together with the system prompt and as much of the history as fits into
   `MAX_CONTEXT_TOKENS` minus `RESPONSE_TOKEN_BUFFER` they form the context.
4. **Generate** – the context is sent to the Bedrock LLM through the Converse
   API, with the system prompt in its own block.

The collection is only read; creating and filling it stays the responsibility
of cvbot-embedder.

## Conversations and context

The full history and the context sent to the model are managed separately:

- A `Conversation` holds every turn unchanged and lives in a
  `ConversationStore` (in-memory for now, a shared backend can be added later).
  This is what a UI displays.
- `build_context` derives a **new**, possibly shorter message list from it. If
  the budget is exceeded, the oldest history messages are dropped one by one;
  the current question is always kept and the system prompt is never part of
  the truncatable list, because it is sent in its own Converse block.
- Tokens are counted with the `cl100k_base` encoding, the same approximation
  cvbot-embedder uses for chunking.
- If the system prompt and the current question alone exceed the budget, the
  call fails instead of silently sending a degraded context.

The system prompt defines the persona (friendly, professional, answers in the
language of the question), forbids inventing facts and contains explicit
prompt-injection guardrails. Retrieved chunks and the question are wrapped in
delimiters and marked as data, and delimiter-like text inside them is
neutralized.

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
| `MAX_CONTEXT_TOKENS` | `8000` | Upper bound for the whole context sent to the LLM |
| `RESPONSE_TOKEN_BUFFER` | `1024` | Part of the budget kept free for the answer |
| `WEB_HOST` | `127.0.0.1` | Interface the web application binds to |
| `WEB_PORT` | `8080` | Port the web application listens on |
| `LOG_LEVEL` | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARNING` or `ERROR` |

`CHROMA_COLLECTION` and `EMBEDDING_MODEL_ID` must match the values used by
cvbot-embedder, otherwise the query vectors are incompatible with the indexed
ones.

## Usage

```python
from cvbot_retriever.config import Settings
from cvbot_retriever.pipeline import ConversationEngine

engine = ConversationEngine(Settings.from_env())
print(engine.answer("conversation-1", "Which projects has the candidate worked on?").answer)
print(engine.answer("conversation-1", "And which technologies were involved?").answer)

for message in engine.store.load("conversation-1").history():
    print(message.role, message.content)
```

`answer_question(settings, question)` remains available for a single question
without history.

For a quick smoke test against a running ChromaDB:

```bash
python -m cvbot_retriever "Which projects has the candidate worked on?" \
  --top-k 6 \
  --log-level DEBUG
```

The command line answers a single question; conversations with several turns
are driven through `ConversationEngine`.

## Web application

The chat UI and the JSON API share the same engine and the same conversation
store:

```bash
python -m cvbot_retriever --serve --host 127.0.0.1 --port 8080
```

Opening `/` creates a conversation and redirects to `/c/<uuid4>`, which renders
the full visible history plus a privacy notice. Only user and assistant turns
are rendered - the system prompt and the retrieved chunks never leave the
process. The page sends questions to the JSON API and therefore needs no build
step.

The engine is built on the first question, not at startup, so the application
comes up while ChromaDB is still unavailable and retries on the next request.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Redirects to a new conversation |
| `GET` | `/c/{conversation_id}` | Chat UI of one conversation |
| `POST` | `/api/conversations` | Hands out a new conversation ID |
| `GET` | `/api/conversations/{id}` | Visible history of a conversation |
| `POST` | `/api/conversations/{id}/messages` | Asks a question |
| `GET` | `/healthz` | Liveness, without touching the backends |

```bash
CID=$(curl -s -X POST localhost:8080/api/conversations | jq -r .conversation_id)
curl -s -X POST "localhost:8080/api/conversations/$CID/messages" \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which projects has the candidate worked on?"}'
```

```json
{
  "conversation_id": "7c1f...",
  "answer": "...",
  "messages": [
    {"role": "user", "content": "Which projects has the candidate worked on?"},
    {"role": "assistant", "content": "..."}
  ]
}
```

Conversation IDs must be UUID4 so that foreign conversations cannot be guessed
through the URL. Failures are answered with `{"detail": "..."}` and a status
code - `400` for an unusable question or ID, `503` when ChromaDB or Bedrock do
not answer, `500` otherwise. The message is meant to be shown to the user as-is;
the underlying exception is only written to the log.

The application is served by a single task instance, so the process-local
`InMemoryConversationStore` is enough to keep concurrent conversations apart.
Turns of one conversation are serialized, different conversations run
concurrently.

## Tests

```bash
python -m pytest
```

The tests run without AWS or network access: Bedrock is replaced by a
deterministic embedding model and a Converse double, ChromaDB by a store
double.

## Deployment

The application runs as a single Fargate task in the cluster provisioned by
cvbot-infra, reachable through an HTTP API Gateway. Three workflows drive it,
all authenticating through GitHub OIDC against `cvbot-gha-deploy-role` - no AWS
access keys are stored in the repository.

| Workflow | Trigger | Effect |
| --- | --- | --- |
| `run-pipeline.yml` | Push to `main` | Runs pytest with coverage into the step summary. Independent of the deployment, and a failing suite fails the job. |
| `deploy.yml` | Push to `main` touching `cvbot_retriever/**`, `requirements.txt`, `Dockerfile`, `.dockerignore` or the workflow itself; or a manual run | Builds the image, pushes it to ECR, registers a new task definition revision and points the service at it. |
| `lifecycle.yml` | Manual only | Toggles the web app and ChromaDB between desired count 0 and 1. |

Documentation-only changes never trigger a deployment.

### Image

`Dockerfile` installs `requirements.txt` into a `python:3.13-slim` image, copies
the package and runs `python -m cvbot_retriever --serve` as an unprivileged user
on port 8080. Only `WEB_HOST` is defaulted to `0.0.0.0` in the image, every
other setting comes from the task definition. The interpreter stays on `PATH`
because the ECS health check calls `/healthz` through `python -c`.

### Rolling out

The image is tagged with the commit SHA, which is what gets deployed, and
additionally with `latest` so that the `webapp_image_tag` default in Terraform
keeps pointing at a real image.

The service is replaced rather than rolled: the running task is drained to 0
before the new revision starts. That preserves the guarantee of exactly one task
which the in-memory conversation store depends on, at the price of a short
outage and lost conversations on every deployment.

A service that is stopped when the deployment runs stays stopped and only gets
pointed at the new revision - starting it costs money and remains a deliberate
manual step.

### Starting and stopping

Both services idle at desired count 0 so that nothing is billed between demos.
Running the "Start or stop web app" workflow with `start` brings ChromaDB up
first and the web app second, then probes `/healthz` through the public
endpoint; `stop` shuts them down in reverse order. They are always toggled
together because the web app cannot answer anything without ChromaDB.

### Repository variables

| Variable | Value |
| --- | --- |
| `AWS_REGION` | `eu-central-1` |
| `AWS_DEPLOY_ROLE_ARN` | `arn:aws:iam::<account-id>:role/cvbot-gha-deploy-role` |
| `ECR_REPOSITORY` | `cvbot-webapp` |
| `ECS_CLUSTER` | `cvbot-cluster` |
| `ECS_WEBAPP_SERVICE` | `cvbot-webapp-service` |
| `ECS_CHROMA_SERVICE` | `cvbot-chroma-service` |
| `ECS_TASK_FAMILY` | `cvbot-webapp` |
| `WEBAPP_API_URL` | Invoke URL of the `cvbot-webapp-api` HTTP API |

The trust policy of the deploy role only accepts OIDC subjects matching the
`gh_oidc_claim` variable in cvbot-infra. It has to cover this repository too,
otherwise `sts:AssumeRoleWithWebIdentity` fails before the first step.

## Layout

```
cvbot_retriever/
  config.py        Settings from environment variables
  embeddings.py    Bedrock embedding model for the question
  vector_store.py  Read-only ChromaDB client and collection
  retriever.py     Top-k chunk lookup
  prompts.py       System prompt and user message construction
  conversation.py  Messages, full history and conversation store
  tokens.py        Token counting for the context budget
  context.py       Truncation of the history to the token budget
  llm.py           Bedrock Converse client for the generation
  pipeline.py      Orchestration of retrieval, context and generation
  schemas.py       Request and response models of the JSON API
  webapp.py        Chat UI and JSON API
  templates/       Jinja2 template of the chat page
  __main__.py      Command line and server start
```

## Known limitations

- Conversations are only kept in memory: they are lost when the process ends
  and are not shared between several application instances.
- Truncation drops whole messages; there is no summarization of older turns.
- Token counting uses `cl100k_base` as an approximation of the Bedrock
  tokenizers, so the real usage can differ slightly.
- The web layer turns Bedrock and ChromaDB failures into a friendly message,
  but does not retry a failed turn; the library itself still propagates them
  unchanged.
- The JSON API has neither rate limiting nor an explicit CORS policy, and
  conversations have no expiry.

