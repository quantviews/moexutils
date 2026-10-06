"""Check public specification/margin sources; retain provenance, never bypass access controls."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import ssl
import urllib.error
import urllib.request

from moexutils import lake

SOURCES = {
    'rts_2022_doc': 'https://www.moex.com/files/4mykz8mc3kyy5wwpgcnppatjnw',
    'br_specification': 'https://fs.moex.com/files/26811',
    'current_margin_xml': 'https://www.moex.com/export/derivatives/go.aspx?type=F',
    'requested_old_margin_xml': 'https://www.moex.com/export/derivatives/go.aspx?type=F&date=2020-01-03',
    'forts_list': 'https://ftp.moex.com/pub/info/stats/forts/FORTS_LIST.TXT',
    'forts_spreads_list': 'https://ftp.moex.com/pub/info/stats/forts/FORTS_LIST_SPREAD.TXT',
    'forts_list_backup': 'https://ftp.moex.com/pub/info/stats/forts/Backup/FORTS_LIST2.TXT',
}


def main():
    root = Path(lake.DATA_ROOT) / 'raw' / 'futures_sources' / dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for name, url in SOURCES.items():
        item = {'name': name, 'source_url': url,
                'observed_at': dt.datetime.now(dt.timezone.utc).isoformat()}
        try:
            # Windows trust store; TLS verification stays enabled.
            with urllib.request.urlopen(url, context=ssl.create_default_context(), timeout=30) as r:
                data = r.read(20_000_001)
                if len(data) > 20_000_000:
                    raise ValueError('Source exceeds 20 MB probe limit')
                path = root / (name + '.bin')
                path.write_bytes(data)
                item.update(status=r.status, content_type=r.headers.get('Content-Type'),
                            final_url=r.url, size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                            file=path.name, last_modified=r.headers.get('Last-Modified'))
        except urllib.error.HTTPError as e:
            item.update(status=e.code, error=str(e))
        except (OSError, ValueError) as e:
            item.update(status=None, error=str(e))
        results.append(item)
        print(name, item.get('status'), item.get('size', 0), flush=True)
    (root / 'manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(root)


if __name__ == '__main__':
    main()
