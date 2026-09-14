# Runtime image of the chat application. The ECS task definition overrides the
# configuration through environment variables and probes /healthz with a plain
# "python -c ...", so the interpreter has to stay on PATH.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WEB_HOST=0.0.0.0 \
    WEB_PORT=8080

WORKDIR /app

# Separate layer so dependency installs are cached across code-only changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY cvbot_retriever/ ./cvbot_retriever/

RUN useradd --system --create-home --shell /usr/sbin/nologin cvbot \
    && chown -R cvbot:cvbot /app
USER cvbot

EXPOSE 8080

CMD ["python", "-m", "cvbot_retriever", "--serve"]
