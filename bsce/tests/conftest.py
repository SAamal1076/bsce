import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

RESULTS = ROOT / "tests" / "results"
RESULTS.mkdir(exist_ok=True)


@pytest.fixture(scope="session")
def record():
    """Collect equivalence numbers; written to tests/results/equivalence.jsonl."""
    rows = []

    def _rec(test, **kw):
        row = {"test": test, **kw}
        rows.append(row)
        print(json.dumps(row))

    yield _rec
    with open(RESULTS / "equivalence.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running full-size equivalence checks")
