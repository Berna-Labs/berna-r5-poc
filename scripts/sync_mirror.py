"""Sync BernaLabs repo to personal mirror.

Usage:
    python scripts/sync_mirror.py
"""

from huggingface_hub import HfApi

SOURCE = "BernaLabs/berna-r5-poc"
TARGET = "muhamedkamil/berna-r5-poc-mirror"


def sync():
    api = HfApi()
    files = list(api.list_repo_files(SOURCE, repo_type="model"))
    print(f"Syncing {len(files)} files: {SOURCE} -> {TARGET}")
    ok, fail = 0, 0
    for f in files:
        try:
            api.upload_file(
                path_or_fileobj=api.hf_hub_download(
                    repo_id=SOURCE, filename=f, repo_type="model",
                ),
                path_in_repo=f,
                repo_id=TARGET,
                repo_type="model",
                commit_message=f"Sync: {f}",
            )
            ok += 1
        except Exception as e:
            print(f"  FAIL: {f} -- {type(e).__name__}")
            fail += 1
    print(f"Done: {ok} OK, {fail} failed")
    print(f"View: https://huggingface.co/{TARGET}")


if __name__ == "__main__":
    sync()
