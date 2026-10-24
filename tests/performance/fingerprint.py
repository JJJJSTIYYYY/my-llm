"""Identify the runtime, fixtures and measuring code used by an offline run."""

import hashlib
from pathlib import Path

import my_llm


def digest(root: Path, paths):
    fingerprint = hashlib.sha256()
    for path in sorted(paths):
        fingerprint.update(path.relative_to(root).as_posix().encode())
        fingerprint.update(b"\0")
        fingerprint.update(path.read_bytes())
        fingerprint.update(b"\0")
    return fingerprint.hexdigest()


def source():
    root = Path(my_llm.__file__).parent
    return digest(root, root.rglob("*.py"))


def harness():
    root = Path(__file__).resolve().parents[2]
    files = [
        *(
            root / "tests/performance" / name
            for name in (
                "fingerprint.py",
                "middleware.py",
                "request_chain.py",
                "runner.py",
                "transport.py",
                "validation.py",
                "workloads.py",
            )
        ),
        *(
            root / "tests/helpers" / name
            for name in (
                "profiles.py",
                "protocols.py",
                "schema.py",
                "sdk.py",
            )
        ),
        *(root / "docs/reference/schema").rglob("*.json"),
    ]
    return digest(root, files)
