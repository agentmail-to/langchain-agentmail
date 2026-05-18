"""Smallest possible example: send one email through an AgentMail inbox.

Required environment variables:
    AGENTMAIL_API_KEY   AgentMail API key (https://www.agentmail.to/)
    AGENTMAIL_INBOX_ID  Inbox to send from
    TO_EMAIL            Recipient address
"""

import os
import sys

from langchain_agentmail import AgentMailClient, AgentMailSendTool


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing required environment variable: {name}")
    return value


def run(client: AgentMailClient, inbox_id: str, to_addr: str) -> str:
    """Send a single hello-world email and return the tool's JSON result."""
    tool = AgentMailSendTool(client=client)
    return tool.invoke(
        {
            "inbox_id": inbox_id,
            "to": to_addr,
            "subject": "Hello from LangChain",
            "text": "This message was sent through langchain-agentmail.",
        }
    )


def main() -> None:
    _require_env("AGENTMAIL_API_KEY")
    inbox_id = _require_env("AGENTMAIL_INBOX_ID")
    to_addr = _require_env("TO_EMAIL")
    print(run(AgentMailClient(), inbox_id, to_addr))


if __name__ == "__main__":
    main()
