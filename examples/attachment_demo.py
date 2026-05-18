"""Send a message with an attached file, then resolve its download URL.

Demonstrates the outbound `SendAttachmentSpec` path (base64-encoded file
bytes) and the inbound `agentmail_get_attachment` flow (presigned download
URL). Run twice with the same inbox to see both halves end-to-end: the
second run can use a `message_id` / `attachment_id` from the first.

Required environment variables:
    AGENTMAIL_API_KEY   AgentMail API key
    AGENTMAIL_INBOX_ID  Inbox to send from
    TO_EMAIL            Recipient address
"""

import base64
import json
import os
import sys

from langchain_agentmail import (
    AgentMailClient,
    AgentMailGetAttachmentTool,
    AgentMailSendTool,
    SendAttachmentSpec,
)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing required environment variable: {name}")
    return value


def send_with_attachment(
    client: AgentMailClient,
    inbox_id: str,
    to_addr: str,
    *,
    file_bytes: bytes = b"hello from langchain-agentmail",
    filename: str = "greeting.txt",
    content_type: str = "text/plain",
) -> dict:
    """Send a message carrying one inline base64 attachment."""
    encoded = base64.b64encode(file_bytes).decode("ascii")
    raw = AgentMailSendTool(client=client).invoke(
        {
            "inbox_id": inbox_id,
            "to": to_addr,
            "subject": "Demo: attachment round-trip",
            "text": "See the attached file.",
            "attachments": [
                SendAttachmentSpec(
                    filename=filename,
                    content_type=content_type,
                    content=encoded,
                )
            ],
        }
    )
    return json.loads(raw)


def download_attachment(
    client: AgentMailClient,
    inbox_id: str,
    message_id: str,
    attachment_id: str,
) -> dict:
    """Resolve the presigned URL for one attachment on a received message."""
    raw = AgentMailGetAttachmentTool(client=client).invoke(
        {
            "inbox_id": inbox_id,
            "message_id": message_id,
            "attachment_id": attachment_id,
        }
    )
    return json.loads(raw)


def main() -> None:
    _require_env("AGENTMAIL_API_KEY")
    inbox_id = _require_env("AGENTMAIL_INBOX_ID")
    to_addr = _require_env("TO_EMAIL")

    sent = send_with_attachment(AgentMailClient(), inbox_id, to_addr)
    print("[sent]", json.dumps(sent, indent=2))
    print(
        "\nTo download the attachment back, grab message_id + attachment_id "
        "(e.g. from agentmail_get_message) and call download_attachment()."
    )


if __name__ == "__main__":
    main()
