"""Smoke tests for the runnable scripts in `examples/`.

Each example exposes a thin `run()` (or named helper) on top of a `main()`
that just reads env vars. The tests drive the helpers with an injected
fake SDK client — no network, no real keys, no LLM calls.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from langchain_agentmail import AgentMailClient

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

import attachment_demo  # noqa: E402
import draft_workflow  # noqa: E402
import rag_inbox  # noqa: E402
import send_one_email  # noqa: E402


def _fake_client(sdk: SimpleNamespace) -> AgentMailClient:
    return AgentMailClient(client=sdk)


# ---------------------------------------------------------------------------
# send_one_email
# ---------------------------------------------------------------------------


def test_send_one_email_run_invokes_send_tool():
    sdk = SimpleNamespace(
        inboxes=SimpleNamespace(
            messages=SimpleNamespace(
                send=MagicMock(return_value={"message_id": "m_1", "thread_id": "t_1"}),
            ),
        ),
    )
    client = _fake_client(sdk)

    out = send_one_email.run(client, "ib_1", "alice@example.com")

    assert '"message_id": "m_1"' in out
    kwargs = sdk.inboxes.messages.send.call_args.kwargs
    assert kwargs["inbox_id"] == "ib_1"
    assert kwargs["to"] == "alice@example.com"
    assert kwargs["subject"] == "Hello from LangChain"
    assert kwargs["text"].startswith("This message was sent")


def test_send_one_email_main_exits_when_env_missing(monkeypatch, capsys):
    for var in ("AGENTMAIL_API_KEY", "AGENTMAIL_INBOX_ID", "TO_EMAIL"):
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(SystemExit) as exc:
        send_one_email.main()

    assert "AGENTMAIL_API_KEY" in str(exc.value)


# ---------------------------------------------------------------------------
# draft_workflow
# ---------------------------------------------------------------------------


def test_draft_workflow_create_update_send_in_order():
    drafts = SimpleNamespace(
        create=MagicMock(return_value={"draft_id": "d_1", "subject": "Draft v1"}),
        update=MagicMock(return_value={"draft_id": "d_1", "subject": "Draft v2 — ready"}),
        send=MagicMock(return_value={"message_id": "m_9", "thread_id": "t_9"}),
    )
    sdk = SimpleNamespace(inboxes=SimpleNamespace(drafts=drafts, messages=SimpleNamespace()))
    client = _fake_client(sdk)

    result = draft_workflow.run(client, "ib_1", "bob@example.com")

    assert result["created"]["draft_id"] == "d_1"
    assert result["revised"]["subject"] == "Draft v2 — ready"
    assert result["sent"]["message_id"] == "m_9"

    drafts.create.assert_called_once()
    create_kwargs = drafts.create.call_args.kwargs
    assert create_kwargs["inbox_id"] == "ib_1"
    assert create_kwargs["to"] == "bob@example.com"
    assert create_kwargs["subject"] == "Draft v1"

    drafts.update.assert_called_once()
    update_kwargs = drafts.update.call_args.kwargs
    assert update_kwargs == {
        "inbox_id": "ib_1",
        "draft_id": "d_1",
        "subject": "Draft v2 — ready",
        "text": "Revised body. Sending now.",
    }

    drafts.send.assert_called_once_with(inbox_id="ib_1", draft_id="d_1")


# ---------------------------------------------------------------------------
# attachment_demo
# ---------------------------------------------------------------------------


def test_attachment_demo_send_serializes_base64_payload():
    sdk = SimpleNamespace(
        inboxes=SimpleNamespace(
            messages=SimpleNamespace(
                send=MagicMock(return_value={"message_id": "m_att", "thread_id": "t_1"}),
            ),
        ),
    )
    client = _fake_client(sdk)

    result = attachment_demo.send_with_attachment(
        client,
        "ib_1",
        "carol@example.com",
        file_bytes=b"payload bytes",
        filename="note.txt",
    )

    assert result == {"message_id": "m_att", "thread_id": "t_1"}
    kwargs = sdk.inboxes.messages.send.call_args.kwargs
    assert kwargs["attachments"] == [
        {
            "filename": "note.txt",
            "content_type": "text/plain",
            "content": base64.b64encode(b"payload bytes").decode("ascii"),
        }
    ]


def test_attachment_demo_download_returns_signed_url():
    sdk = SimpleNamespace(
        inboxes=SimpleNamespace(
            messages=SimpleNamespace(
                get_attachment=MagicMock(
                    return_value={
                        "attachment_id": "a_1",
                        "filename": "note.txt",
                        "download_url": "https://example.com/signed",
                    }
                ),
            ),
        ),
    )
    client = _fake_client(sdk)

    result = attachment_demo.download_attachment(client, "ib_1", "m_1", "a_1")

    assert result["download_url"] == "https://example.com/signed"
    sdk.inboxes.messages.get_attachment.assert_called_once_with(
        inbox_id="ib_1", message_id="m_1", attachment_id="a_1"
    )


# ---------------------------------------------------------------------------
# rag_inbox
# ---------------------------------------------------------------------------


def _rag_sdk(messages: list[dict], full_by_id: dict[str, dict]) -> AgentMailClient:
    sdk = SimpleNamespace(
        inboxes=SimpleNamespace(
            messages=SimpleNamespace(
                list=MagicMock(return_value={"messages": messages}),
                get=MagicMock(side_effect=lambda inbox_id, message_id: full_by_id[message_id]),
            ),
        ),
    )
    return _fake_client(sdk)


def test_rag_keyword_search_returns_scored_docs():
    client = _rag_sdk(
        messages=[{"message_id": "m_1"}, {"message_id": "m_2"}, {"message_id": "m_3"}],
        full_by_id={
            "m_1": {"message_id": "m_1", "subject": "invoice for april", "text": "see attached"},
            "m_2": {"message_id": "m_2", "subject": "hello", "text": "talked about invoice once"},
            "m_3": {"message_id": "m_3", "subject": "spam", "text": "no match"},
        },
    )
    docs = rag_inbox.keyword_search(client, "ib", "invoice", k=2)
    assert len(docs) == 2
    assert docs[0].metadata["message_id"] == "m_1"


def test_rag_load_documents_respects_limit():
    client = _rag_sdk(
        messages=[{"message_id": f"m_{i}"} for i in range(5)],
        full_by_id={f"m_{i}": {"message_id": f"m_{i}", "text": "x"} for i in range(5)},
    )
    docs = rag_inbox.load_documents(client, "ib", limit=3)
    assert len(docs) == 3
