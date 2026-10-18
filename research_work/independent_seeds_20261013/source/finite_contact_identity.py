"""Explicit source lineage for reviewed, lossless experiment recovery."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def active_identity(root):
    root=Path(root)
    plan=json.loads((root/'pipeline.json').read_text())
    name=plan.get('source_identity_file','source_identity_before.json')
    if name!='source_identity_before.json':
        review=json.loads((root/'checks/memory_recovery_review.json').read_text())
        assert review['status']=='passed' and review['source_identity_file']==name
        assert review['source_identity_sha256']==digest(root/name)
    identity=json.loads((root/name).read_text())
    assert all(digest(ROOT/path)==sha for path,sha in identity.items()),'Active training source changed'
    return identity
