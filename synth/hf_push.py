"""Push generated route outputs to a Hugging Face dataset repo instead of local disk.

Each route lands in its OWN FOLDER on the repo: <route_name>/<route_name>.json, so a dev
can pull just the routes they want. The dataset artifact never touches local disk — it is
serialized in memory and uploaded. (The per-route STATE checkpoint under synth/state/ is
kept locally regardless: it is the resume ledger, not the deliverable.)

Auth: HF_API_KEY (env or ROOT/.env), reusing engine's .env reader. Requires
huggingface_hub (present in the nanochat venv the pipeline runs under).
"""

import io
import json
import os
import sys
from pathlib import Path

# Standalone (no engine import, so it stays free of the httpx transport dep and runs in
# any env that has huggingface_hub).
ROOT = Path(__file__).resolve().parent.parent
HF_REPO = "zachnorton03/synthetic-pre1930-sft"
HF_REPO_TYPE = "dataset"


def _from_dotenv(name):
    """Read NAME from ROOT/.env (KEY=value or `export KEY=value`), else None."""
    f = ROOT / ".env"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        line = line[len("export "):].strip() if line.startswith("export ") else line
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'")
    return None


def _token():
    tok = os.environ.get("HF_API_KEY") or _from_dotenv("HF_API_KEY")
    if not tok:
        sys.exit("Set HF_API_KEY (env or ROOT/.env) to push to Hugging Face.")
    return tok


def _api():
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError:
        sys.exit("huggingface_hub not installed in this env. Install it into the venv "
                 "you run under, e.g.  uv pip install --python .venv/bin/python huggingface_hub")
    return HfApi(token=_token())


def push_rows(route_name, rows, repo=HF_REPO):
    """Upload `rows` (a list of dicts) as <route_name>/<route_name>.json to the HF
    dataset repo, creating the repo if it does not yet exist. Returns the repo path."""
    api = _api()
    api.create_repo(repo_id=repo, repo_type=HF_REPO_TYPE, exist_ok=True)
    data = json.dumps(rows, indent=2, ensure_ascii=False).encode("utf-8")
    path_in_repo = f"{route_name}/{route_name}.json"
    api.upload_file(
        path_or_fileobj=io.BytesIO(data),
        path_in_repo=path_in_repo,
        repo_id=repo,
        repo_type=HF_REPO_TYPE,
        commit_message=f"Update {route_name} ({len(rows)} rows)",
    )
    return f"{repo}/{path_in_repo}"


def test_connection(repo=HF_REPO):
    """Validate auth + write access without leaving junk: upload a tiny probe file,
    confirm it lands, then delete it. Prints the outcome and returns True on success."""
    from datetime import datetime
    api = _api()
    api.create_repo(repo_id=repo, repo_type=HF_REPO_TYPE, exist_ok=True)
    probe = f".connection_test/probe_{datetime.now():%Y%m%d_%H%M%S}.json"
    payload = json.dumps({"ok": True, "at": datetime.now().isoformat()}).encode("utf-8")
    api.upload_file(path_or_fileobj=io.BytesIO(payload), path_in_repo=probe,
                    repo_id=repo, repo_type=HF_REPO_TYPE,
                    commit_message="connection test (temporary probe)")
    files = api.list_repo_files(repo_id=repo, repo_type=HF_REPO_TYPE)
    landed = probe in files
    if landed:
        api.delete_file(path_in_repo=probe, repo_id=repo, repo_type=HF_REPO_TYPE,
                        commit_message="remove connection-test probe")
    print(f"HF connection to {repo}: {'OK (probe uploaded, verified, removed)' if landed else 'FAILED'}")
    return landed


if __name__ == "__main__":
    test_connection()
