#!/usr/bin/env python

import sys
from csv import DictReader
from pathlib import Path
from tempfile import gettempdir
from typing import Dict


def main():
    assert len(sys.argv) == 2, "Usage: parse_csv.py <csv_file>"

    file = sys.argv[1]
    row: Dict[str, str] = dict()
    with open(Path(gettempdir()) / "full" / "{file.replace('::', '_')}.csv", "r") as f:
        reader = DictReader(f)

        for row in reader:
            pass

    print(f".beta_range({row['beta']}, {2 * float(row['beta'])})")

    print(".initial_densities([")
    for i, (k, v) in enumerate(filter(lambda kv: kv[0].startswith("n-"), row.items())):
        print(f"({i}, {v}),")
    print("])")
    print(".initial_asymmetries([")
    for i, (k, v) in enumerate(filter(lambda kv: kv[0].startswith("na-"), row.items())):
        print(f"({i}, {v}),")
    print("])")


if __name__ == "__main__":
    main()
