"""Permanent observations and changes, outside DuckLake retention."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4
from moexutils import lake


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, default=str, indent=2)+'\n', encoding='utf-8')


def new_run():
    now = datetime.now(timezone.utc)
    path = Path(lake.DATA_ROOT) / 'versions/openpositions' / (now.strftime('%Y%m%dT%H%M%S%fZ')+'-'+uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    save_json(path/'run.json', dict(started_at=now.isoformat(), published_at=None,
                                  semantics='local observation, not historical publication'))
    return path


def changes(old, new, keys):
    before = {tuple(r[k] for k in keys): r for r in old.to_dicts()}
    after = {tuple(r[k] for k in keys): r for r in new.to_dicts()}
    return [dict(operation='insert' if k not in before else 'delete' if k not in after else 'update',
                 key=dict(zip(keys,k)), before=before.get(k), after=after.get(k))
            for k in before.keys() | after.keys() if before.get(k) != after.get(k)]


def finish(path, errors):
    save_json(path/'result.json', dict(finished_at=datetime.now(timezone.utc).isoformat(),
                                      complete=not errors, errors=[str(e) for e in errors]))
    save_json(path/'manifest.json', dict(artifacts={p.relative_to(path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in sorted(path.rglob('*')) if p.is_file()}))
