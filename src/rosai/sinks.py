"""Optional LangSmith upload. Nothing is sent unless a key is set and upload is on."""

import os

DEFAULT_MAX_TRACES = 5000


def upload_traces(
    examples,
    client=None,
    *,
    upload: bool = False,
    max_traces: int = DEFAULT_MAX_TRACES,
) -> int:
    key = os.environ.get("LANGSMITH_API_KEY")
    if not upload or not key:
        return 0
    if client is None:
        from langsmith import Client

        client = Client(api_key=key)
    chosen = list(examples)[:max_traces]
    for example in chosen:
        client.create_run(**example)
    return len(chosen)
