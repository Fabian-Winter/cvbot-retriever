"""End-to-end tests of the pipeline and the command line."""

from __future__ import annotations

import pytest

from cvbot_retriever import __main__ as cli
from cvbot_retriever import pipeline
from cvbot_retriever.config import Settings
from cvbot_retriever.llm import BedrockLLMClient
from tests.conftest import FakeBedrockRuntime, FakeEmbeddings, FakeStore, make_documents

CHUNKS = make_documents(
    "The candidate studied computer science in Karlsruhe.",
    "Since 2020 the candidate works as a platform engineer.",
)


@pytest.fixture
def patched_pipeline(
    monkeypatch: pytest.MonkeyPatch, fake_embeddings: FakeEmbeddings
) -> FakeBedrockRuntime:
    """Replaces AWS and ChromaDB access with test doubles.

    Args:
        monkeypatch: pytest fixture for temporarily replacing attributes.
        fake_embeddings: Deterministic embedding model.

    Returns:
        The Bedrock runtime the pipeline generates the answer with.
    """
    runtime = FakeBedrockRuntime(["The candidate studied computer science."])
    monkeypatch.setattr(pipeline, "build_embeddings", lambda s: fake_embeddings)
    monkeypatch.setattr(pipeline, "create_client", lambda s: object())
    monkeypatch.setattr(
        pipeline, "open_collection", lambda c, name, emb: FakeStore(CHUNKS)
    )
    monkeypatch.setattr(
        pipeline,
        "BedrockLLMClient",
        lambda s: BedrockLLMClient(s, client=runtime),
    )
    return runtime


def test_answer_question_returns_generated_answer(
    settings: Settings, patched_pipeline: FakeBedrockRuntime
) -> None:
    result = pipeline.answer_question(settings, "What did the candidate study?")

    assert result.question == "What did the candidate study?"
    assert result.answer == "The candidate studied computer science."
    assert result.chunks == CHUNKS


def test_answer_question_grounds_the_prompt_in_the_chunks(
    settings: Settings, patched_pipeline: FakeBedrockRuntime
) -> None:
    pipeline.answer_question(settings, "What did the candidate study?")

    [request] = patched_pipeline.calls
    prompt = request["messages"][0]["content"][0]["text"]
    assert "What did the candidate study?" in prompt
    assert all(chunk.page_content in prompt for chunk in CHUNKS)


def test_answer_question_retrieves_top_k_chunks(
    settings: Settings, patched_pipeline: FakeBedrockRuntime
) -> None:
    result = pipeline.answer_question(
        settings.with_overrides(top_k=1), "What did the candidate study?"
    )

    assert result.chunks == CHUNKS[:1]


def test_answer_question_rejects_empty_question(
    settings: Settings, patched_pipeline: FakeBedrockRuntime
) -> None:
    with pytest.raises(ValueError, match="question"):
        pipeline.answer_question(settings, "  ")


def test_build_prompt_without_chunks_still_contains_the_question() -> None:
    prompt = pipeline.build_prompt("Anything?", [])

    assert "Anything?" in prompt


def test_cli_prints_the_answer(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    patched_pipeline: FakeBedrockRuntime,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: settings))

    exit_code = cli.main(["What did the candidate study?", "--top-k", "1"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "The candidate studied computer science."


def test_cli_reports_failures(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(cli.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(
        cli, "answer_question", lambda s, q: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    assert cli.main(["Any question?"]) == 1
