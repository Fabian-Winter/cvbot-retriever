# cvbot-retriever

Retrieval and generation for a RAG chatbot: it embeds a question with AWS
Bedrock, looks up the matching document chunks in the ChromaDB filled by
cvbot-embedder and lets a Bedrock LLM generate an answer from them - across
several turns of a conversation.

## How it works

1. **Condense and extract** – one small model call rewrites the question into
   a standalone search query and extracts metadata filters from it. Both come
   back as a single JSON object, so the second step costs no extra call.
2. **Embed** – the query is embedded with the same Bedrock model the chunks
   were indexed with.
3. **Retrieve** – `TOP_K * OVERFETCH_FACTOR` nearest chunks are read from the
   ChromaDB collection together with their vector distance, including their
   metadata (`source`, `filename`, `chunk_index` and the section metadata
   written by cvbot-embedder). They are then re-ranked by similarity plus a
   filter bonus and a recency bonus (see below) and cut back to `TOP_K`.
4. **Build the context** – chunks and question become the current user message;
   together with the system prompt and as much of the history as fits into
   `MAX_CONTEXT_TOKENS` minus `RESPONSE_TOKEN_BUFFER` they form the context.
5. **Generate** – the context is sent to the Bedrock LLM through the Converse
   API, with the system prompt in its own block.

The collection is only read; creating and filling it stays the responsibility
of cvbot-embedder.

## Retrieval ranking

Every candidate is scored by three additive parts, and the best `TOP_K` win:

- **Similarity** – the vector distance mapped onto `(0, 1]` via `1 / (1 + d)`,
  so a closer chunk always contributes more.
- **Filter bonus** – up to `FILTER_WEIGHT`, scaled by the share of the
  extracted metadata fields the chunk satisfies. The bonus is capped, so a
  chunk never outranks a much closer one through field count alone.
- **Recency bonus** – `RECENCY_WEIGHT` scaled by how recently the section
  ended, read at query time from the `startdate`/`enddate`/`status` metadata
  cvbot-embedder already wrote: an open-ended `enddate` (`now`, `laufend`, absent)
  or `status: current` counts as the present, a concrete one decays linearly to
  zero over `RECENCY_WINDOW_YEARS`. A section without any date stays neutral,
  which keeps the undated documents competitive. `RECENCY_WEIGHT=0` turns the
  bonus off and restores the pure similarity order. No re-indexing is needed.

`OVERFETCH_FACTOR` controls how far down the similarity ranking a boosted chunk
may still win from.

## Metadata filters

The filterable schema is not configured here: cvbot-embedder publishes the
fields it observed in the documents on the collection, and it is injected into
the condensation prompt so the model knows what it may filter on.

The pipeline is deliberately tolerant, in both directions:

- **Nothing is ever excluded.** Filters only re-rank. A matching chunk gains up
  to `FILTER_WEIGHT`, scaled by the share of the extracted fields it satisfies,
  and moves up; a chunk that lacks the field keeps its place. A filter without
  a single match therefore degrades into plain semantic search instead of
  returning nothing.
- **Nothing is ever invented.** Every extracted field and value is validated
  against the published schema; anything unknown is dropped with a log entry.
  A question without a filterable criterion yields empty filters, not an error.
- **Nothing is ever required.** A collection indexed before this feature, or
  one whose documents carry no metadata, reports an empty schema – the whole
  step then behaves exactly as before, including the free first turn.

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
| `LLM_MODEL_ID` | `eu.amazon.nova-2-lite-v1:0` | Bedrock model ID for the answer |
| `TOP_K` | `4` | Number of chunks retrieved per question |
| `OVERFETCH_FACTOR` | `4` | How many times `TOP_K` is fetched before similarity, filters and recency re-rank the candidates |
| `FILTER_WEIGHT` | `0.2` | Largest score the filter bonus adds, scaled by the share of matching metadata fields, relative to the similarity score of 0 to 1 |
| `RECENCY_WEIGHT` | `0.2` | Largest score the recency bonus adds; `0` turns it off |
| `RECENCY_WINDOW_YEARS` | `10` | How many years back the recency bonus decays to zero |
| `MAX_CONTEXT_TOKENS` | `8000` | Upper bound for the whole context sent to the LLM |
| `RESPONSE_TOKEN_BUFFER` | `1024` | Part of the budget kept free for the answer |
| `WEB_HOST` | `127.0.0.1` | Interface the web application binds to |
| `WEB_PORT` | `8080` | Port the web application listens on |
| `LOG_LEVEL` | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARNING` or `ERROR` |
| `RATE_LIMIT_PER_MINUTE` | `10` | Answered questions per client and minute |
| `RATE_LIMIT_PER_HOUR` | `60` | Answered questions per client and hour |
| `TRUST_FORWARDED_FOR` | `true` | Read the client address from `X-Forwarded-For` |
| `CORS_ALLOWED_ORIGINS` | empty | Comma separated origins allowed to call the JSON API |
| `CONVERSATION_TTL_SECONDS` | `1800` | Idle time after which a conversation is dropped |
| `MAX_CONVERSATIONS` | `50` | Conversations kept in memory at once |

`CHROMA_COLLECTION` must match the value used by cvbot-embedder. The embedding
model itself is not configured here: cvbot-retriever reads the model ID from
the metadata cvbot-embedder stored on the collection and embeds the question
with that exact model.

## Security

The application is publicly reachable, so it is hardened against abuse, cost
run-away and unnecessary data retention.

### Rate limiting

Every answer costs one Bedrock embedding call and one Bedrock LLM call, which
makes an unthrottled endpoint a direct cost risk. `POST /api/conversations` and
`POST /api/conversations/{id}/messages` are therefore limited per client, with
a sliding minute and hour window; the stricter of the two wins. A client over
its budget receives `429` with a `Retry-After` header and a friendly message,
and the engine is never touched. Reading endpoints and `/healthz` stay
unthrottled, so the chat page and the ECS health check keep working while a
client is blocked.

A client is identified by the first entry of `X-Forwarded-For`, falling back to
the peer address. Behind the API Gateway every request arrives through the same
VPC link interface, so the peer address alone would put all callers into one
bucket. The header is only read when `TRUST_FORWARDED_FOR` is set, because a
direct caller could forge it; it is enabled in AWS, where the API Gateway is the
only way in, and should be disabled when the application is exposed directly.

This complements the API Gateway throttling (`webapp_throttle_rate_limit`),
which caps the total load but cannot tell clients apart.

### CORS

The JSON API sets an explicit policy. `CORS_ALLOWED_ORIGINS` is empty by
default, which emits no CORS headers at all and keeps the API same-origin; the
bundled UI is served from the same origin and is unaffected. Configured origins
must be complete origins (`https://example.com`) - a wildcard, a missing scheme
or a path is rejected when the configuration is loaded. Credentials are never
allowed, and only `GET` and `POST` are.

### Prompt injection

The system prompt marks the retrieved documents and the question as data rather
than instructions, and refuses to disclose its own configuration. Retrieved
chunks and questions are additionally stripped of the delimiter sequences that
frame those blocks, so user input cannot close them early.

### Conversation data

Conversations live in the memory of the single task and are never written to
disk or to a database. An entry that was idle for `CONVERSATION_TTL_SECONDS` is
dropped on the next access, and the store keeps at most `MAX_CONVERSATIONS`
entries, dropping the least recently used one beyond that. Both bounds limit
how long conversation content exists at all and keep a long running task from
growing with every visitor. A restart discards everything.

### Logs

Logs carry metadata only: counts, model IDs, conversation IDs and error types.
Questions and answers are never logged, not even when a request is rejected -
the validation handler logs the failing field names instead of the Pydantic
error objects, which would contain the full user input. CloudWatch retention is
capped in Terraform (`log_retention_days`, seven days by default).

### Outside the scope of this proof of concept

- No authentication or authorization: the application is intentionally public
  and holds no data that is not meant to be public.
- No WAF, bot detection or CAPTCHA in front of the API Gateway.
- The rate limit is process-local. It is sound for the single task this service
  runs as, but would need a shared store if it were ever scaled out.
- A client behind a shared NAT counts as one client.
- `/api/docs` stays publicly reachable; it describes the API surface but
  exposes no internals.
- No encryption of conversations in memory, and no audit trail of who asked
  what.

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
  retriever.py     Candidate fetch with over-fetching
  ranking.py       Re-ranking by similarity, boost and recency
  query_condensation.py  Standalone query and metadata filter extraction
  prompts.py       System prompt and user message construction
  conversation.py  Messages, full history and conversation store
  tokens.py        Token counting for the context budget
  context.py       Truncation of the history to the token budget
  llm.py           Bedrock Converse client for the generation
  pipeline.py      Orchestration of retrieval, context and generation
  schemas.py       Request and response models of the JSON API
  ratelimit.py     Per-client rate limiting of the public endpoints
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

