"""Stage a draft, revise it, then send.

Drafts are the right tool when an agent wants to compose iteratively (or
schedule a future delivery via `send_at`). This example walks through the
three verbs the SDK exposes: `create`, `update`, `send`.

Required environment variables:
    AGENTMAIL_API_KEY   AgentMail API key
    AGENTMAIL_INBOX_ID  Inbox to send from
    TO_EMAIL            Recipient address
"""

import json
import os
import sys

from langchain_agentmail import (
    AgentMailClient,
    AgentMailCreateDraftTool,
    AgentMailSendDraftTool,
    AgentMailUpdateDraftTool,
)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing required environment variable: {name}")
    return value


def run(client: AgentMailClient, inbox_id: str, to_addr: str) -> dict:
    """Create -> update -> send a draft. Returns parsed JSON of each step."""
    create = AgentMailCreateDraftTool(client=client)
    update = AgentMailUpdateDraftTool(client=client)
    send = AgentMailSendDraftTool(client=client)

    created = json.loads(
        create.invoke(
            {
                "inbox_id": inbox_id,
                "to": to_addr,
                "subject": "Draft v1",
                "text": "First pass — will be revised before sending.",
            }
        )
    )
    draft_id = created["draft_id"]

    revised = json.loads(
        update.invoke(
            {
                "inbox_id": inbox_id,
                "draft_id": draft_id,
                "subject": "Draft v2 — ready",
                "text": "Revised body. Sending now.",
            }
        )
    )

    sent = json.loads(send.invoke({"inbox_id": inbox_id, "draft_id": draft_id}))
    return {"created": created, "revised": revised, "sent": sent}


def main() -> None:
    _require_env("AGENTMAIL_API_KEY")
    inbox_id = _require_env("AGENTMAIL_INBOX_ID")
    to_addr = _require_env("TO_EMAIL")
    result = run(AgentMailClient(), inbox_id, to_addr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
