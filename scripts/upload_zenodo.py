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
    ap.add_argument("--deposition", default=None)
    ap.add_argument("--file", default="_local/zenodo/zenodo_v3.1/alpacavision-v3.1.zip")
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

    if a.deposition:
        draft = check(requests.get(f"{API}/deposit/depositions/{a.deposition}", params=p,
                                   timeout=60), "get draft").json()
    else:
        rec = check(requests.get(f"{API}/records/{a.record}", timeout=60), "resolve record").json()
        r = check(requests.post(f"{API}/deposit/depositions/{rec['id']}/actions/newversion",
                                params=p, timeout=60), "newversion")
        draft = check(requests.get(r.json()["links"]["latest_draft"], params=p, timeout=60),
                      "get draft").json()
    dep_id, bucket = draft["id"], draft["links"]["bucket"]
    reserved = draft["metadata"].get("prereserve_doi", {}).get("doi")
    print(f"draft id : {dep_id}   reserved DOI: {reserved}")

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
                       params=p, json={"metadata": {**meta, "prereserve_doi": True}},
                       headers={"Content-Type": "application/json"}, timeout=60),
          "metadata")
    print("metadata set")

    print(f"\nDRAFT READY, NOT PUBLISHED")
    print(f"  review it at: https://zenodo.org/uploads/{dep_id}")
    print("  check the files and metadata, then press Publish there.")
    print("  publishing mints the new DOI and cannot be undone.")


if __name__ == "__main__":
    main()
