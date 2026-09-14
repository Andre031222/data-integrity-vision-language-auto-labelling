#!/usr/bin/env python3
"""Upload a new version of the dataset to Zenodo and leave it as an unpublished draft.

The token is read from ZENODO_TOKEN in the environment or in .env, and is never
printed. Publishing is deliberately not automated: a Zenodo record cannot be
unpublished, so the final button is left to a human.
"""
import argparse, json, os, sys
from pathlib import Path

import requests

API = "https://zenodo.org/api"


def load_token():
    tok = os.environ.get("ZENODO_TOKEN")
    if tok:
        return tok.strip()
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ZENODO_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("ZENODO_TOKEN not found. Put it in .env as ZENODO_TOKEN=... "
             "or export it. Create one at zenodo.org/account/settings/applications/tokens/new "
             "with the scopes deposit:write and deposit:actions.")


def check(r, what):
    if r.status_code >= 400:
        sys.exit(f"{what} failed [{r.status_code}]: {r.text[:400]}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="21134001",
                    help="Zenodo concept or version id; the latest version is resolved")
    ap.add_argument("--file", default="_local/zenodo_v3/alpacavision-v3.0.zip")
    ap.add_argument("--metadata", default="scripts/zenodo_metadata.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would happen without touching Zenodo")
    a = ap.parse_args()

    src = Path(a.file)
    if not src.exists():
        sys.exit(f"file not found: {src}")
    meta = json.loads(Path(a.metadata).read_text())["metadata"]

    print(f"file     : {src}  ({src.stat().st_size / 2**20:.1f} MiB)")
    print(f"record   : {a.record}")
    print(f"title    : {meta['title']}")
    print(f"version  : {meta.get('version')}")
    print(f"licence  : {meta.get('license')}")
    if a.dry_run:
        print("\ndry run: nothing sent.")
        return

    tok = load_token()
    p = {"access_token": tok}

    # A concept id and a version id look alike; only the version id can be edited.
    rec = check(requests.get(f"{API}/records/{a.record}", timeout=60), "resolve record").json()
    version_id = rec["id"]
    if str(version_id) != str(a.record):
        print(f"resolved : concept {a.record} -> latest version {version_id}")
    print(f"concept DOI: {rec.get('conceptdoi')}  (this is the one to cite)")

    print("\ncreating a new version...")
    r = check(requests.post(f"{API}/deposit/depositions/{version_id}/actions/newversion",
                            params=p, timeout=60), "newversion")
    draft_url = r.json()["links"]["latest_draft"]
    draft = check(requests.get(draft_url, params=p, timeout=60), "get draft").json()
    dep_id, bucket = draft["id"], draft["links"]["bucket"]
    print(f"draft id : {dep_id}")

    for f in draft.get("files", []):
        requests.delete(f"{API}/deposit/depositions/{dep_id}/files/{f['id']}",
                        params=p, timeout=60)
    print(f"inherited files removed: {len(draft.get('files', []))}")

    print("uploading (this takes a while)...")
    with src.open("rb") as fh:
        check(requests.put(f"{bucket}/{src.name}", data=fh, params=p, timeout=3600),
              "upload")
    print("upload done")

    check(requests.put(f"{API}/deposit/depositions/{dep_id}",
                       params=p, json={"metadata": meta},
                       headers={"Content-Type": "application/json"}, timeout=60),
          "metadata")
    print("metadata set")

    print(f"\nDRAFT READY, NOT PUBLISHED")
    print(f"  review it at: https://zenodo.org/uploads/{dep_id}")
    print("  check the files and metadata, then press Publish there.")
    print("  publishing mints the new DOI and cannot be undone.")


if __name__ == "__main__":
    main()
