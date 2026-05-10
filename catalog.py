"""
Builds and caches the SHL Individual Test Solutions catalog.
On first run, attempts to scrape from shl.com; falls back to a
bundled snapshot when the live site is unreachable or returns no results.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

CATALOG_PATH = Path("catalog.json")
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

# ── Fallback catalog (real SHL Individual Test Solutions) ─────────────────────
FALLBACK_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "OPQ32 (Occupational Personality Questionnaire)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32r/",
        "description": (
            "Measures 32 personality characteristics across three domains: "
            "relationships with people, thinking styles, and feelings and emotions. "
            "Used for managerial, professional, and graduate-level roles to predict "
            "leadership potential, team fit, and cultural alignment."
        ),
        "tags": ["personality", "leadership", "managerial", "professional", "behavioural"],
    },
    {
        "name": "Motivation Questionnaire (MQ)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/motivation-questionnaire-mq/",
        "description": (
            "Assesses 18 dimensions of workplace motivation including energy and "
            "dynamism, commercial focus, and personal growth. Helps predict engagement, "
            "retention risk, and job satisfaction for any level or function."
        ),
        "tags": ["motivation", "engagement", "behavioural", "retention"],
    },
    {
        "name": "Verify G+ (General Ability)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/general-ability/",
        "description": (
            "Adaptive test measuring overall cognitive ability (g-factor) by combining "
            "numerical, verbal, and inductive reasoning. Strong predictor of learning "
            "potential and performance across a wide range of roles."
        ),
        "tags": ["cognitive", "general ability", "adaptive", "numerical", "verbal", "inductive"],
    },
    {
        "name": "Verify Numerical Reasoning",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/numerical-reasoning-1/",
        "description": (
            "Measures ability to analyse and interpret numerical data presented as "
            "tables, graphs, and charts. Suitable for roles requiring data analysis, "
            "finance, engineering, or any job where numerical judgement is critical."
        ),
        "tags": ["cognitive", "numerical", "data analysis", "finance", "analytical"],
    },
    {
        "name": "Verify Verbal Reasoning",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verbal-reasoning-1/",
        "description": (
            "Assesses ability to understand, evaluate, and draw conclusions from "
            "written information. Essential for roles requiring strong communication, "
            "policy analysis, consulting, legal, or HR work."
        ),
        "tags": ["cognitive", "verbal", "communication", "analytical", "reading"],
    },
    {
        "name": "Verify Inductive Reasoning",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/inductive-reasoning-1/",
        "description": (
            "Tests abstract reasoning by asking candidates to identify rules and "
            "patterns in sequences of shapes. Predicts problem-solving ability and "
            "adaptability to new situations, especially for technical and IT roles."
        ),
        "tags": ["cognitive", "inductive", "abstract", "problem-solving", "technical", "IT"],
    },
    {
        "name": "Verify Deductive Reasoning",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/deductive-reasoning-1/",
        "description": (
            "Measures ability to draw logical conclusions from provided premises and "
            "rules. Suited for roles requiring structured analysis, compliance, legal, "
            "or operations where rule-based thinking is important."
        ),
        "tags": ["cognitive", "deductive", "logical", "analytical", "compliance"],
    },
    {
        "name": "Verify Mechanical Comprehension",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/mechanical-comprehension-1/",
        "description": (
            "Evaluates understanding of mechanical principles such as levers, pulleys, "
            "gears, and fluid dynamics. Used for engineering, trades, manufacturing, "
            "and technical maintenance positions."
        ),
        "tags": ["cognitive", "mechanical", "engineering", "technical", "trades", "manufacturing"],
    },
    {
        "name": "Verify Calculation",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/calculation/",
        "description": (
            "Measures speed and accuracy in basic arithmetic calculations including "
            "addition, subtraction, multiplication, and division. Relevant for clerical, "
            "financial, and operational roles requiring numerical accuracy."
        ),
        "tags": ["cognitive", "numerical", "calculation", "clerical", "accuracy", "speed"],
    },
    {
        "name": "Verify Checking",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/checking/",
        "description": (
            "Assesses attention to detail by asking candidates to spot errors or "
            "mismatches in data sets. Critical for data-entry, administrative, "
            "quality control, and financial accuracy roles."
        ),
        "tags": ["cognitive", "attention to detail", "clerical", "accuracy", "data entry"],
    },
    {
        "name": "Automata Fix (Coding Bug-Fix Assessment)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/automata-fix/",
        "description": (
            "Presents candidates with broken code and asks them to identify and fix "
            "bugs. Tests practical debugging skills in multiple languages (Java, Python, "
            "JavaScript, C++, SQL). Ideal for developers at all levels."
        ),
        "tags": ["coding", "debugging", "software", "technical", "Java", "Python", "JavaScript", "C++", "SQL"],
    },
    {
        "name": "Automata Pro (Full Coding Simulation)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/automata-pro/",
        "description": (
            "Full-stack coding simulation requiring candidates to build or extend a "
            "real feature in a realistic IDE environment. Measures end-to-end "
            "software engineering skill including design, coding, and testing."
        ),
        "tags": ["coding", "software engineering", "full-stack", "senior", "technical"],
    },
    {
        "name": "Verify Interactive Java",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-java/",
        "description": (
            "Adaptive coding assessment specifically for Java developers. Covers "
            "object-oriented design, collections, concurrency, exception handling, "
            "and Java-specific APIs. Suitable for junior to senior Java roles."
        ),
        "tags": ["coding", "Java", "software", "technical", "OOP", "backend"],
    },
    {
        "name": "Verify Interactive Python",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-python/",
        "description": (
            "Adaptive coding assessment for Python developers. Covers data structures, "
            "algorithms, OOP, file handling, and Python standard library usage. "
            "Relevant for data engineers, ML engineers, and backend developers."
        ),
        "tags": ["coding", "Python", "software", "technical", "backend", "data engineering", "ML"],
    },
    {
        "name": "Verify Interactive SQL",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-sql/",
        "description": (
            "Assesses SQL proficiency including joins, subqueries, aggregations, "
            "window functions, and query optimisation. Used for data analysts, "
            "database administrators, and backend developers."
        ),
        "tags": ["coding", "SQL", "database", "data analysis", "technical", "DBA"],
    },
    {
        "name": "Verify Interactive JavaScript",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-javascript/",
        "description": (
            "Adaptive coding test for JavaScript developers covering DOM manipulation, "
            "async programming, closures, prototypal inheritance, and modern ES6+ "
            "features. Suitable for frontend and Node.js roles."
        ),
        "tags": ["coding", "JavaScript", "frontend", "Node.js", "web development", "technical"],
    },
    {
        "name": "Verify Interactive C++",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-cpp/",
        "description": (
            "Evaluates C++ programming skills including memory management, pointers, "
            "templates, STL, and performance optimisation. Used for systems "
            "programming, embedded, and game development roles."
        ),
        "tags": ["coding", "C++", "systems", "embedded", "technical", "performance"],
    },
    {
        "name": "Verify Interactive .NET/C#",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/verify-interactive-dot-net/",
        "description": (
            "Coding assessment for C# and .NET developers. Covers LINQ, async/await, "
            "dependency injection, ASP.NET, and .NET runtime concepts. Relevant "
            "for enterprise backend and cloud-native .NET roles."
        ),
        "tags": ["coding", "C#", ".NET", "backend", "enterprise", "technical"],
    },
    {
        "name": "Contact Center Simulation",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/contact-center-simulation/",
        "description": (
            "Realistic simulation of customer service and contact centre scenarios "
            "including live-chat, email, and call-handling tasks. Measures empathy, "
            "multitasking, accuracy, and customer-first communication."
        ),
        "tags": ["simulation", "customer service", "contact centre", "communication", "operational"],
    },
    {
        "name": "Customer Service Assessment",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/customer-service/",
        "description": (
            "Measures customer-facing competencies such as empathy, communication "
            "clarity, problem resolution, and patience. Suitable for retail, "
            "hospitality, support, and service desk positions."
        ),
        "tags": ["customer service", "behavioural", "communication", "retail", "hospitality"],
    },
    {
        "name": "Sales Achievement Predictor",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/sales-achievement-predictor/",
        "description": (
            "Combines personality and motivational data to predict sales performance. "
            "Covers competitiveness, persuasion, resilience, and drive. Validated "
            "for field sales, inside sales, and account management roles."
        ),
        "tags": ["sales", "personality", "motivation", "behavioural", "commercial"],
    },
    {
        "name": "Situational Judgement Test (SJT)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/situational-judgement/",
        "description": (
            "Presents realistic workplace scenarios and asks candidates to choose the "
            "most and least effective responses. Assesses practical judgement, "
            "values alignment, and decision-making at all career levels."
        ),
        "tags": ["situational judgement", "behavioural", "decision-making", "judgement", "values"],
    },
    {
        "name": "Graduate Item Bank (GIB)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/graduate-item-bank/",
        "description": (
            "High-difficulty cognitive battery (numerical, verbal, inductive) "
            "calibrated for graduate and early-career professional recruitment. "
            "Differentiates high-potential graduates from large applicant pools."
        ),
        "tags": ["cognitive", "graduate", "early career", "numerical", "verbal", "inductive"],
    },
    {
        "name": "Managerial/Professional Item Bank (MIB/PIB)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/managerial-professional-item-bank/",
        "description": (
            "Cognitive battery at managerial-level difficulty: numerical, verbal, "
            "and inductive reasoning. Benchmarked against senior professional and "
            "management populations worldwide."
        ),
        "tags": ["cognitive", "managerial", "professional", "senior", "numerical", "verbal"],
    },
    {
        "name": "Dependability and Safety Instrument (DSI)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/dependability-and-safety-instrument/",
        "description": (
            "Assesses counterproductive work behaviours, safety compliance attitudes, "
            "and reliability. Used for safety-critical, blue-collar, and operational "
            "roles where risk and dependability matter most."
        ),
        "tags": ["safety", "dependability", "behavioural", "operational", "blue collar", "risk"],
    },
    {
        "name": "Work Strengths Assessment",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/work-strengths/",
        "description": (
            "Measures practical work readiness competencies for entry-level positions: "
            "reliability, flexibility, teamwork, and customer focus. Designed for "
            "high-volume frontline and hourly worker hiring."
        ),
        "tags": ["entry level", "frontline", "behavioural", "competency", "high volume", "hourly"],
    },
    {
        "name": "Technology Professional Assessments",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/technology-professional/",
        "description": (
            "Broad battery for IT professionals covering logical reasoning, "
            "technology concepts, and role-specific knowledge modules. "
            "Applicable to IT support, systems administration, and tech consulting."
        ),
        "tags": ["IT", "technology", "cognitive", "technical", "systems", "consulting"],
    },
    {
        "name": "Administrative Professional Assessments",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/administrative-professional/",
        "description": (
            "Measures clerical speed, accuracy, and administrative competencies "
            "including data entry, document checking, and business writing. "
            "Suited for PA, executive assistant, and office administration roles."
        ),
        "tags": ["administrative", "clerical", "accuracy", "data entry", "office", "PA"],
    },
    {
        "name": "Universal Competency Framework (UCF) 360",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/ucf-360/",
        "description": (
            "360-degree multi-rater assessment based on SHL's Universal Competency "
            "Framework covering 20 competency clusters. Used for leadership development, "
            "succession planning, and performance management."
        ),
        "tags": ["360", "leadership", "development", "competency", "managerial", "succession"],
    },
    {
        "name": "Scenarios (Personality-based Situational Assessment)",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/scenarios/",
        "description": (
            "Immersive scenario-based assessment combining personality measurement "
            "with situational judgement. Candidates make decisions in branching "
            "scenarios; scores map to OPQ personality scales for deeper insight."
        ),
        "tags": ["personality", "situational judgement", "behavioural", "immersive", "senior"],
    },
    {
        "name": "Smart Interview Live",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/smart-interview-live/",
        "description": (
            "AI-scored live structured interview solution. Provides behavioural "
            "interview questions aligned to competencies, real-time scoring guidance, "
            "and post-interview analytics for hiring managers."
        ),
        "tags": ["interview", "structured interview", "behavioural", "AI scoring", "managerial"],
    },
    {
        "name": "Smart Interview On Demand",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/smart-interview-on-demand/",
        "description": (
            "Video interview platform where candidates record responses to structured "
            "questions asynchronously. AI analyses verbal content, communication clarity, "
            "and response structure to rank candidates at scale."
        ),
        "tags": ["interview", "video interview", "asynchronous", "AI scoring", "high volume"],
    },
]


async def _scrape_catalog(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    """Attempt to scrape Individual Test Solutions from shl.com."""
    assessments: List[Dict[str, Any]] = []

    try:
        response = await client.get(
            CATALOG_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; SHLBot/1.0)"},
            timeout=20,
            follow_redirects=True,
        )
        response.raise_for_status()
    except Exception as exc:
        logger.warning("Could not fetch SHL catalog page: %s", exc)
        return assessments

    soup = BeautifulSoup(response.text, "lxml")

    # SHL renders a table/grid of products; look for product links
    product_links = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if "/product-catalog/view/" in href:
            full_url = href if href.startswith("http") else f"https://www.shl.com{href}"
            if full_url not in product_links:
                product_links.append(full_url)

    if not product_links:
        logger.warning("No product links found on catalog page — site may block scraping.")
        return assessments

    logger.info("Found %d product links, fetching detail pages…", len(product_links))

    for url in product_links[:40]:  # cap to avoid rate-limiting
        try:
            detail = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; SHLBot/1.0)"},
                timeout=15,
                follow_redirects=True,
            )
            detail.raise_for_status()
            dsoup = BeautifulSoup(detail.text, "lxml")

            # Extract title
            title_tag = dsoup.find("h1") or dsoup.find("h2")
            name = title_tag.get_text(strip=True) if title_tag else url.split("/")[-2]

            # Extract description from meta or first meaningful paragraph
            meta_desc = dsoup.find("meta", attrs={"name": "description"})
            if meta_desc and meta_desc.get("content"):
                description = meta_desc["content"].strip()
            else:
                paragraphs = dsoup.find_all("p")
                description = " ".join(
                    p.get_text(strip=True) for p in paragraphs[:3] if p.get_text(strip=True)
                )[:500]

            if name and description:
                assessments.append(
                    {
                        "name": name,
                        "url": url,
                        "description": description,
                        "tags": [],
                    }
                )
                logger.debug("Scraped: %s", name)
            await asyncio.sleep(0.5)  # polite delay
        except Exception as exc:
            logger.debug("Failed to fetch %s: %s", url, exc)

    return assessments


async def build_catalog() -> List[Dict[str, Any]]:
    """
    Returns catalog: loads from disk if present, else tries scraping,
    then falls back to the bundled snapshot.
    """
    if CATALOG_PATH.exists():
        logger.info("Loading catalog from %s", CATALOG_PATH)
        with open(CATALOG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
        if data:
            return data

    logger.info("Catalog not found — attempting to scrape shl.com…")
    async with httpx.AsyncClient() as client:
        scraped = await _scrape_catalog(client)

    if scraped:
        logger.info("Scraped %d assessments from shl.com", len(scraped))
        catalog = scraped
    else:
        logger.info("Scraping yielded no results — using bundled fallback catalog (%d items)", len(FALLBACK_CATALOG))
        catalog = FALLBACK_CATALOG

    with open(CATALOG_PATH, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=2, ensure_ascii=False)
    logger.info("Catalog saved to %s", CATALOG_PATH)
    return catalog
