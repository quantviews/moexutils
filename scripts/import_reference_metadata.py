"""Build a versioned metadata release and optionally append it to DuckLake."""
import argparse
import json
from pathlib import Path

from moexutils import lake, reference_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("reference_metadata_sources.json")))
    parser.add_argument("--data-root", default=lake.DATA_ROOT)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    path = reference_metadata.build(args.config, args.data_root)
    print(json.dumps(dict(package=str(path), published=reference_metadata.publish(path) if args.publish else False)))
