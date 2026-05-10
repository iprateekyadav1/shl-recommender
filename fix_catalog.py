"""
Fetch real SHL catalog, repair malformed JSON (bare newlines inside strings),
parse it, and write catalog.json.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
OUT = Path(__file__).resolve().parent / "catalog.json"


def fix_json_newlines(data: bytes) -> bytes:
    """Replace bare \\n / \\r inside JSON string values with escape sequences."""
    result = bytearray()
    in_string = False
    escape_next = False
    for b in data:
        if escape_next:
            result.append(b)
            escape_next = False
        elif b == ord("\\"):
            result.append(b)
            escape_next = True
        elif b == ord('"'):
            result.append(b)
            in_string = not in_string
        elif in_string and b == 0x0A:   # bare LF inside string
            result.extend(b"\\n")
        elif in_string and b == 0x0D:   # bare CR inside string
            result.extend(b"\\r")
        else:
            result.append(b)
    return bytes(result)


def main():
    print("Fetching catalog...", flush=True)
    with urllib.request.urlopen(URL, timeout=30) as resp:
        raw = resp.read()
    print(f"Downloaded {len(raw):,} bytes", flush=True)

    fixed = fix_json_newlines(raw)
    print(f"Fixed size: {len(fixed):,} bytes", flush=True)

    try:
        data = json.loads(fixed)
    except json.JSONDecodeError as exc:
        print(f"Still failing at pos {exc.pos}: {exc.msg}")
        ctx = fixed[max(0, exc.pos - 120): exc.pos + 120]
        print("Context:", ctx)
        sys.exit(1)

    print(f"Parsed OK — type: {type(data).__name__}", flush=True)

    # Explore structure
    if isinstance(data, list):
        items = data
        print(f"Top-level list with {len(items)} items")
        if items:
            print("First item keys:", list(items[0].keys()))
    elif isinstance(data, dict):
        print("Top-level dict keys:", list(data.keys())[:15])
        # Try common wrapper keys
        for key in ("products", "items", "catalog", "data", "assessments"):
            if key in data:
                items = data[key]
                print(f"Found '{key}' with {len(items)} items")
                if items:
                    print("First item keys:", list(items[0].keys()))
                break
        else:
            # Print structure for manual inspection
            for k, v in data.items():
                print(f"  {k!r}: {type(v).__name__} len={len(v) if hasattr(v,'__len__') else 'N/A'}")
            items = []
    else:
        print("Unexpected type:", type(data))
        items = []

    if not items:
        print("No items found — aborting.")
        sys.exit(1)

    # Show sample item
    print("\nSample item:")
    print(json.dumps(items[0], indent=2)[:1000])
    print(f"\nTotal items: {len(items)}")

    # Save raw parsed data for inspection
    with open(OUT.parent / "catalog_raw.json", "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"Saved raw to catalog_raw.json")


if __name__ == "__main__":
    main()
