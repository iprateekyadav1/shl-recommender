"""Quick eval: check if key expected items appear in recommendations for sample conversation queries."""
import json
import urllib.request

BASE = "http://127.0.0.1:8000"

def chat(messages):
    body = json.dumps({"messages": messages}).encode()
    req = urllib.request.Request(BASE + "/chat", data=body,
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

# (query, expected_name_substrings)
TESTS = [
    (
        "We need a solution for senior leadership. CXOs, director level. Selection - comparing against a leadership benchmark.",
        ["OPQ32r", "OPQ Leadership Report"],
    ),
    (
        "Hiring a senior Rust engineer for high-performance networking. Should I add cognitive test?",
        ["Linux Programming", "Networking and Implementation", "Verify Interactive G+"],
    ),
    (
        "Screening 500 entry-level contact centre agents. Inbound calls. English US.",
        ["Contact Center Call Simulation", "Customer Service Phone Simulation"],
    ),
    (
        "Hiring graduate financial analysts. Need numerical reasoning and finance knowledge test.",
        ["Numerical Reasoning", "Financial Accounting"],
    ),
    (
        "Hiring plant operators for a chemical facility. Safety is absolute top priority.",
        ["Dependability and Safety Instrument", "Safety"],
    ),
    (
        "I need to quickly screen admin assistants for Excel and Word.",
        ["Excel", "Word"],
    ),
    (
        "Senior Full-Stack Engineer: Java, Spring, REST API, SQL, AWS, Docker. Backend-leaning.",
        ["Java", "Spring", "SQL"],
    ),
]

passed = 0
total = 0

for query, expected in TESTS:
    try:
        resp = chat([{"role": "user", "content": query}])
        rec_names = [r["name"] for r in resp.get("recommendations", [])]
        rec_str = " | ".join(rec_names)
        hits = [e for e in expected if any(e.lower() in n.lower() for n in rec_names)]
        miss = [e for e in expected if e not in hits]
        status = "PASS" if not miss else "PARTIAL"
        passed += len(hits)
        total += len(expected)
        print(f"{status}  [{len(hits)}/{len(expected)}] {query[:60]}")
        if miss:
            print(f"       Missing: {miss}")
        print(f"       Got: {rec_str[:100]}")
    except Exception as exc:
        print(f"ERROR  {query[:60]}: {exc}")
    print()

print(f"\nScore: {passed}/{total} ({100*passed//total}%)")
