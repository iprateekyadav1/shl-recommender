"""
Build catalog.json from the raw SHL catalog data already in catalog_raw.json.
Maps SHL 'keys' categories to single/multi-letter test_type codes.
"""
import json
import re
from pathlib import Path

RAW = Path(__file__).resolve().parent / "catalog_raw.json"
OUT = Path(__file__).resolve().parent / "catalog.json"

KEY_TO_CODE = {
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
    "Ability & Aptitude": "A",
    "Competencies": "C",
    "Biodata & Situational Judgment": "B",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}


def keys_to_test_type(keys_list):
    if not keys_list:
        return "K"
    codes = []
    seen = set()
    for k in keys_list:
        c = KEY_TO_CODE.get(k)
        if c and c not in seen:
            codes.append(c)
            seen.add(c)
    return ",".join(codes) if codes else "K"


def make_tags(item):
    tags = []

    # From test type keys
    for k in (item.get("keys") or []):
        tags.append(k.lower().replace(" & ", " ").replace(" ", "-"))

    # From job levels
    level_map = {
        "Director": "director leadership senior",
        "Executive": "executive leadership senior",
        "Manager": "manager managerial",
        "Front Line Manager": "frontline manager supervisor",
        "Supervisor": "supervisor frontline",
        "Graduate": "graduate early-career",
        "Entry-Level": "entry-level frontline hourly",
        "Mid-Professional": "mid-level professional",
        "Professional Individual Contributor": "professional individual-contributor",
        "General Population": "general",
    }
    for lvl in (item.get("job_levels") or []):
        mapped = level_map.get(lvl, lvl.lower())
        for word in mapped.split():
            if word not in tags:
                tags.append(word)

    # From name — extract keywords
    name = item.get("name", "")
    name_lower = name.lower()

    tech_keywords = [
        "java", "python", "sql", "javascript", "c++", "c#", ".net",
        "spring", "angular", "react", "aws", "docker", "kubernetes",
        "linux", "networking", "excel", "word", "powerpoint", "office",
        "salesforce", "sap", "html", "css", "php", "ruby", "scala",
        "r programming", "tableau", "power bi", "git", "devops",
        "azure", "google cloud", "hadoop", "spark", "tensorflow",
        "hipaa", "accounting", "finance", "statistics", "mathematics",
        "medical", "healthcare", "retail", "contact center", "call center",
        "customer service", "sales", "marketing", "hr", "project management",
        "agile", "scrum", "testing", "qa", "security", "cybersecurity",
    ]
    for kw in tech_keywords:
        if kw in name_lower or kw in (item.get("description") or "").lower():
            clean = kw.replace(" ", "-")
            if clean not in tags:
                tags.append(clean)

    # Adaptive / remote flags
    if item.get("adaptive") == "yes":
        tags.append("adaptive")
    if item.get("remote") == "yes":
        tags.append("remote")

    # Duration signal
    dur = item.get("duration", "") or ""
    if dur:
        try:
            mins = int(re.search(r"\d+", dur).group())
            if mins <= 15:
                tags.append("short")
            elif mins <= 30:
                tags.append("medium")
            else:
                tags.append("long")
        except Exception:
            pass

    return list(dict.fromkeys(tags))  # deduplicate preserving order


def main():
    with open(RAW, encoding="utf-8") as f:
        raw_items = json.load(f)

    print(f"Loaded {len(raw_items)} raw items")

    catalog = []
    skipped = 0

    for item in raw_items:
        name = (item.get("name") or "").strip()
        url = (item.get("link") or "").strip()
        description = (item.get("description") or "").strip()

        if not name or not url:
            skipped += 1
            continue

        if item.get("status") not in ("ok", None, ""):
            skipped += 1
            continue

        test_type = keys_to_test_type(item.get("keys") or [])
        tags = make_tags(item)

        # Build a rich description if we have duration/languages
        desc_parts = [description]
        if item.get("duration"):
            desc_parts.append(f"Duration: {item['duration']}.")
        if item.get("languages"):
            langs = item["languages"]
            if len(langs) <= 5:
                desc_parts.append(f"Available in: {', '.join(langs)}.")
            else:
                desc_parts.append(
                    f"Available in {len(langs)} languages including {', '.join(langs[:3])} and more."
                )

        full_desc = " ".join(p for p in desc_parts if p)

        catalog.append({
            "name": name,
            "url": url,
            "description": full_desc,
            "test_type": test_type,
            "tags": tags,
            "job_levels": item.get("job_levels") or [],
            "duration": item.get("duration") or "",
            "languages": item.get("languages") or [],
            "adaptive": item.get("adaptive") == "yes",
            "remote": item.get("remote") == "yes",
        })

    print(f"Catalog built: {len(catalog)} items (skipped {skipped})")

    # Show distribution
    from collections import Counter
    type_counts = Counter()
    for c in catalog:
        for t in c["test_type"].split(","):
            type_counts[t.strip()] += 1
    print("Test type distribution:", dict(type_counts.most_common()))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"Saved to {OUT}")

    # Show some key items the evaluator checks
    key_names = [
        "Occupational Personality Questionnaire OPQ32r",
        "SHL Verify Interactive G+",
        "Graduate Scenarios",
        "Core Java (Advanced Level) (New)",
        "SQL (New)",
        "Dependability and Safety Instrument (DSI)",
    ]
    print("\nKey items check:")
    name_map = {c["name"]: c for c in catalog}
    for n in key_names:
        if n in name_map:
            c = name_map[n]
            print(f"  ✓ {c['name']} | {c['test_type']} | {c['url']}")
        else:
            print(f"  ✗ NOT FOUND: {n}")


if __name__ == "__main__":
    main()
