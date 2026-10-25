# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Writes a small synthetic GVF dataset: a demo of the training pipeline, NOT real training data.

Every template is its own group, so a grouped validation split only shows the model templates it has never seen.
Run: python benchmarks/synthetic_dataset.py <output directory> [--variants N]
"""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import numpy as np

VULNERABLE = {
    "VC-1": [
        'def find(cursor, {v}):\n    query = "SELECT * FROM {t} WHERE id = " + {v}\n    cursor.execute(query)\n',
        'import os\n\ndef ping({v}):\n    os.system("ping -c 1 " + {v})\n',
        'import subprocess\n\ndef listing({v}):\n    subprocess.call("ls " + {v}, shell=True)\n',
    ],
    "VC-2": [
        "import pickle\n\ndef restore({v}):\n    return pickle.loads({v})\n",
        "import yaml\n\ndef parse({v}):\n    return yaml.load({v})\n",
    ],
    "VC-3": [
        'API_KEY = "{secret}"\n',
        'import db\n\npassword = "{secret}"\ndb.connect(user="admin", password=password)\n',
    ],
}

CLEAN = [
    'def find(cursor, {v}):\n    cursor.execute("SELECT * FROM {t} WHERE id = %s", ({v},))\n',
    'import subprocess\n\ndef ping({v}):\n    subprocess.run(["ping", "-c", "1", {v}])\n',
    "import json\n\ndef restore({v}):\n    return json.loads({v})\n",
    "import yaml\n\ndef parse({v}):\n    return yaml.safe_load({v})\n",
    'import os\n\nAPI_KEY = os.environ["API_KEY"]\n',
    'greeting = "hello {t}"\nprint(greeting)\n',
]

NAMES = [("user_id", "users"), ("payload", "orders"), ("value", "items")]
_IDENTIFIER_WORDS = ["user", "data", "item", "record", "value", "payload", "entry", "name", "token", "row"]


def template_count() -> int:
    return sum(len(templates) for templates in VULNERABLE.values()) + len(CLEAN)


def write_dataset(directory: Path, variants: int = len(NAMES), seed: int = 0) -> Path:
    """Writes variants files per template plus manifest.jsonl, and returns the manifest path."""
    rng = np.random.default_rng(seed)
    alphabet = list(string.ascii_letters + string.digits)
    templates = [(t, [code]) for code, ts in VULNERABLE.items() for t in ts] + [(t, []) for t in CLEAN]
    entries = []
    for template_index, (template, labels) in enumerate(templates):
        for variant in range(variants):
            variable, table = NAMES[variant] if variant < len(NAMES) else _random_names(rng)
            secret = "".join(rng.choice(alphabet, size=32))
            relative = f"sample_{template_index}_{variant}.py"
            (directory / relative).write_text(template.format(v=variable, t=table, secret=secret), encoding="utf-8")
            entries.append(json.dumps({"path": relative, "labels": labels, "group": f"template-{template_index}"}))
    manifest = directory / "manifest.jsonl"
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return manifest


def _random_names(rng: np.random.Generator) -> tuple[str, str]:
    first, second = rng.choice(_IDENTIFIER_WORDS, size=2, replace=False)
    return f"{first}_{second}", f"{second}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--variants", type=int, default=len(NAMES), help="files per template")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    arguments.directory.mkdir(parents=True, exist_ok=True)
    manifest = write_dataset(arguments.directory, arguments.variants, arguments.seed)
    print(f"{template_count() * arguments.variants} files, {template_count()} groups -> {manifest.as_posix()}")


if __name__ == "__main__":
    main()
