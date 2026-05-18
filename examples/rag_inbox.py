"""Search an inbox two ways: keyword retriever, then a semantic vector store.

The keyword path uses `AgentMailRetriever` and needs nothing beyond this
package. The semantic path is opt-in — it pipes `AgentMailLoader().load()`
into an in-memory vector store and answers a question with an LLM. It runs
only when `OPENAI_API_KEY` is set and `langchain-openai` is installed
(pull in via the `examples` extra:
`pip install 'langchain-agentmail[examples]'`).

Required environment variables:
    AGENTMAIL_API_KEY   AgentMail API key
    AGENTMAIL_INBOX_ID  Inbox to read from
    OPENAI_API_KEY      Optional — enables the semantic path
"""

from __future__ import annotations

import os
import sys

from langchain_core.documents import Document

from langchain_agentmail import (
    AgentMailClient,
    AgentMailLoader,
    AgentMailRetriever,
)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"Missing required environment variable: {name}")
    return value


def keyword_search(
    client: AgentMailClient,
    inbox_id: str,
    query: str,
    *,
    k: int = 3,
) -> list[Document]:
    """Run a keyword retriever against the inbox. No embeddings needed."""
    retriever = AgentMailRetriever(client=client, inbox_id=inbox_id, k=k)
    return retriever.invoke(query)


def load_documents(
    client: AgentMailClient,
    inbox_id: str,
    *,
    limit: int = 50,
) -> list[Document]:
    """Load up to `limit` messages as LangChain Documents — ready to embed."""
    return AgentMailLoader(inbox_id=inbox_id, client=client, limit=limit).load()


def _semantic_answer(docs: list[Document], query: str) -> str | None:
    """Embed docs, retrieve top-k, ask an LLM to summarize. Optional path."""
    try:
        from langchain_core.vectorstores import InMemoryVectorStore
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        return None

    store = InMemoryVectorStore.from_documents(docs, OpenAIEmbeddings())
    hits = store.as_retriever(search_kwargs={"k": 3}).invoke(query)
    context = "\n\n---\n\n".join(d.page_content for d in hits)
    response = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(
        f"Use the following emails to answer the question.\n\n"
        f"Emails:\n{context}\n\nQuestion: {query}"
    )
    return str(response.content)


def main() -> None:
    _require_env("AGENTMAIL_API_KEY")
    inbox_id = _require_env("AGENTMAIL_INBOX_ID")

    client = AgentMailClient()
    query = "invoice"

    print("== keyword retriever ==")
    for doc in keyword_search(client, inbox_id, query, k=3):
        subject = doc.metadata.get("subject") or "(no subject)"
        sender = doc.metadata.get("from") or "(unknown)"
        print(f"  - {subject!r} from {sender}")

    if not os.environ.get("OPENAI_API_KEY"):
        print("\n(set OPENAI_API_KEY to also run the semantic path)")
        return

    print("\n== semantic retriever + LLM ==")
    docs = load_documents(client, inbox_id, limit=50)
    if not docs:
        print("  inbox empty — nothing to index")
        return
    answer = _semantic_answer(docs, query)
    if answer is None:
        print("  install langchain-openai to enable semantic search")
        return
    print(answer)


if __name__ == "__main__":
    main()
