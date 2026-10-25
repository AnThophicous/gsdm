# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""python -m dataset_tools.cvefixes {prepare,census} ..."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from dataset_tools.cvefixes.census import format_report, run_census
from dataset_tools.cvefixes.prepare import PrepareError, prepare_from_dump, prepare_from_zip, verify_md5
from dataset_tools.cvefixes.schema import SchemaError


def main() -> int:
    arguments = _parser().parse_args()
    try:
        return arguments.handler(arguments)
    except (PrepareError, SchemaError, sqlite3.Error, OSError) as error:
        print(f"cvefixes: error: {error}", file=sys.stderr)
        return 2


def _prepare(arguments: argparse.Namespace) -> int:
    started = time.perf_counter()
    if arguments.zip:
        if not arguments.skip_md5:
            print("checking MD5 against the Zenodo record...", flush=True)
            verify_md5(arguments.zip)
        statements = prepare_from_zip(arguments.zip, arguments.output, sys.stdout)
    else:
        statements = prepare_from_dump(arguments.dump, arguments.output, sys.stdout)
    print(f"done: {statements:,} statements in {time.perf_counter() - started:.0f} s -> {arguments.output}")
    return 0


def _census(arguments: argparse.Namespace) -> int:
    if not arguments.database.is_file():
        raise OSError(f"database not found: {arguments.database}")
    connection = sqlite3.connect(f"{arguments.database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        report = run_census(connection)
    finally:
        connection.close()
    print(format_report(report))
    if arguments.report:
        arguments.report.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"\nJSON report: {arguments.report}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dataset_tools.cvefixes")
    commands = parser.add_subparsers(required=True)

    prepare = commands.add_parser("prepare", help="import the CVEfixes SQL dump into a SQLite database")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", type=Path, help="the release zip from Zenodo (read without extracting)")
    source.add_argument("--dump", type=Path, help="an already extracted .sql or .sql.gz dump")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--skip-md5", action="store_true")
    prepare.set_defaults(handler=_prepare)

    census = commands.add_parser("census", help="count the usable GVF examples without building anything")
    census.add_argument("database", type=Path)
    census.add_argument("--report", type=Path, help="also write the report as JSON")
    census.set_defaults(handler=_census)
    return parser


if __name__ == "__main__":
    sys.exit(main())
