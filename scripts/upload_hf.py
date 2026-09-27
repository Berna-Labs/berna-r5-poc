"""Upload Berna R5 to HuggingFace Hub.

Usage:
    python scripts/upload_hf.py --repo USER/berna-r5-poc
    python scripts/upload_hf.py --repo USER/berna-r5-poc --private

Requires HF_TOKEN environment variable or huggingface-cli login.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path("/data/berna-r5")


def check_auth() -> str:
    """Verify HuggingFace authentication."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token
        token = get_token()
        if token:
            return token
    except ImportError:
        pass
    # Fallback to old API
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            return token
    except ImportError:
        pass
    print("ERROR: No HF token found.")
    print("  Set HF_TOKEN env var, or run: hf auth login")
    sys.exit(1)


def collect_files() -> list:
    """List of files to upload (relative to ROOT)."""
    files = [
        "README.md",
        "MODEL_CARD.md",
        "LICENSE",
        "requirements.txt",
        "pyproject.toml",
        ".gitignore",
        # Docs
        "docs/00_concept.md",
        "docs/03_deployment.md",
        "docs/zero_error_creed.md",
        # Theory
        "theory/01_saturation.md",
        "theory/02_reliable_mutation.md",
        "theory/03_zero_forgetting.md",
        "theory/04_balanced_growth.md",
        "theory/05_neural_connectivity.md",
        "theory/argmax_convergence.md",
        "theory/justification_100_chromosomes.md",
        # Proofs
        "theory/proofs/README.md",
        "theory/proofs/assumptions.md",
        "theory/proofs/T1_saturation_equilibrium.md",
        "theory/proofs/T2_incorporation_consistency.md",
        "theory/proofs/T3_zero_forgetting.md",
        "theory/proofs/T4_balanced_growth.md",
        "theory/proofs/T5_connectivity_convergence.md",
        # Paper
        "paper/00_outline.md",
        "paper/01_introduction.md",
        "paper/02_related_work.md",
        "paper/03_architecture.md",
        "paper/04_theory.md",
        "paper/05_experiments.md",
        "paper/06_discussion_conclusion_limitations.md",
        # Experiments
        "experiments/00_plan.md",
        # Registry schema (not the .db)
        "registry/schema.sql",
        # Source
        "src/__init__.py",
        "src/config.py",
        "src/zero_error.py",
        "src/registry.py",
        "src/saturation.py",
        "src/plexus.py",
        "src/cell.py",
        "src/chromosome.py",
        "src/dna_kernel.py",
        "src/incorporation.py",
        "src/keys.py",
        "src/deployment.py",
        "src/training.py",
        "src/incorporation_demo.py",
        "src/growth_demo.py",
        "src/experiment_runner.py",
    ]
    return files


def verify_files(files: list) -> tuple:
    """Split into existing and missing."""
    existing = []
    missing = []
    for f in files:
        p = ROOT / f
        if p.exists():
            existing.append(f)
        else:
            missing.append(f)
    return existing, missing


# Files renamed during upload (HF conventions)
RENAME_MAP = {
    "MODEL_CARD.md": "README.md",       # HF uses README.md as model card
    "README.md": "PROJECT_README.md",   # our project readme renamed
}


def upload(repo_id: str, private: bool = False,
           dry_run: bool = False) -> None:
    """Upload all files to HuggingFace Hub."""
    check_auth()

    files = collect_files()
    existing, missing = verify_files(files)

    print(f"Files to upload: {len(existing)}")
    if missing:
        print(f"Missing (skipped): {len(missing)}")
        for m in missing:
            print(f"  - {m}")

    if dry_run:
        print("\nDRY RUN: no upload performed.")
        print("\nFiles that would be uploaded:")
        for f in existing:
            size = (ROOT / f).stat().st_size
            remote = RENAME_MAP.get(f, f)
            marker = f" -> {remote}" if remote != f else ""
            print(f"  {f}  ({size:,} bytes){marker}")
        return

    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    print(f"\nCreating repo: {repo_id} (private={private})")
    try:
        create_repo(repo_id=repo_id, private=private,
                    repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Note: create_repo returned: {e}")

    print(f"\nUploading {len(existing)} files...")
    for f in existing:
        local = ROOT / f
        remote = RENAME_MAP.get(f, f)
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=remote,
                repo_id=repo_id,
                repo_type="model",
            )
            marker = " (renamed)" if remote != f else ""
            print(f"  OK: {f} -> {remote}{marker}")
        except Exception as e:
            print(f"  FAIL: {f} -- {e}")

    print(f"\nDone. View at: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload Berna R5 to HuggingFace Hub"
    )
    parser.add_argument("--repo", required=True,
                        help="HF repo ID (e.g., user/berna-r5-poc)")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repository")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without uploading")
    args = parser.parse_args()

    print("=" * 60)
    print("Berna R5 -- HuggingFace Upload")
    print("=" * 60)
    print(f"Repo: {args.repo}")
    print(f"Private: {args.private}")
    print(f"Dry run: {args.dry_run}")

    upload(args.repo, private=args.private, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
