"""Read-only: print the structure of data/pages/inventory.json. Changes nothing."""
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "data" / "pages" / "inventory.json"


def short(v, n=70):
    s = repr(v)
    return s if len(s) <= n else s[:n] + "..."


def describe(obj, indent=0, depth=6):
    pad = "  " * indent
    if isinstance(obj, dict):
        print(f"{pad}dict with {len(obj)} keys")
        if depth == 0:
            return
        for i, (k, v) in enumerate(obj.items()):
            if i >= 10:
                print(f"{pad}  ... {len(obj) - 10} more keys")
                break
            if isinstance(v, (dict, list)):
                print(f"{pad}  {k}:")
                describe(v, indent + 2, depth - 1)
            else:
                print(f"{pad}  {k}: {type(v).__name__} = {short(v)}")
    elif isinstance(obj, list):
        print(f"{pad}list of {len(obj)}")
        if obj and depth > 0:
            print(f"{pad}  first item:")
            describe(obj[0], indent + 2, depth - 1)
    else:
        print(f"{pad}{type(obj).__name__} = {short(obj)}")


data = json.loads(PATH.read_text(encoding="utf-8"))
print(f"file: {PATH} ({PATH.stat().st_size:,} bytes)")
describe(data)