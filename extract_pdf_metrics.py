# -*- coding: utf-8 -*-
"""Extract evaluation-metric-related text from Desktop PDFs."""
import re
import sys
from pathlib import Path

DESKTOP = Path(r"c:\Users\Zhang\Desktop")

# User-requested files (base names / patterns)
TARGETS = [
    "3319535.3363224.pdf",
    "2503.02702v2.pdf",
    "222.pdf",
    "1710.00811v2.pdf",
    "1710.00811v2_translated.pdf",
    "111.pdf",
    "1-s2.0-S0925231221018646-main.pdf",
    "333.pdf",
    "978-3-030-56223-6_15.pdf",
    "2109.02568v1.pdf",
]

def find_pdf_by_substring(sub: str) -> Path | None:
    for p in DESKTOP.glob("*.pdf"):
        if sub.lower() in p.name.lower():
            return p
    return None

def resolve_paths() -> list[tuple[str, Path]]:
    """Map each user-listed paper to one Path; group duplicates by same resolved path."""
    rules: list[tuple[str, str]] = [
        ("3319535.3363224", "3319535.3363224"),
        ("2503.02702v2", "2503.02702v2"),
        ("222.pdf", "222.pdf"),
        ("1710.00811v2", "1710.00811v2.pdf"),
        ("1710.00811v2_translated", "1710.00811v2_translated"),
        ("111.pdf", "111.pdf"),
        ("ITDBERT", "ITDBERT"),
        ("S0925231221018646", "S0925231221018646"),
        ("Hierarchical_NTPP", "Insider_Threat_Detection_via_Hierarchical"),
        ("LaAeb_S0167404824004310", "S0167404824004310"),
        ("333.pdf", "333.pdf"),
        ("978-3-030-56223-6_15", "978-3-030-56223-6_15"),
        ("Multi-AE long filename", "Insider Threat Detection Using Multi-autoencoder"),
        ("Dual-Domain_GCN", "Dual-Domain_Graph_Convolutional"),
        ("2109.02568v1", "2109.02568v1"),
        ("2509.01509v1", "2509.01509v1.pdf"),
        ("applsci-09-04018-v2", "applsci-09-04018-v2.pdf"),
        ("S0167404821000456", "1-s2.0-S0167404821000456-main.pdf"),
        ("S0167404825002767", "1-s2.0-S0167404825002767-main.pdf"),
        ("2509.01509v1_translated", "2509.01509v1_translated.pdf"),
    ]
    out: list[tuple[str, Path]] = []
    seen_paths: set[Path] = set()
    for label, sub in rules:
        p = DESKTOP / sub if sub.endswith(".pdf") and (DESKTOP / sub).exists() else None
        if p is None:
            p = find_pdf_by_substring(sub.replace(".pdf", ""))
        if p is None:
            print(f"MISSING: {label} ({sub})", file=sys.stderr)
            continue
        out.append((label, p))
        seen_paths.add(p.resolve())
    return out


def extract_pypdf(path: Path) -> str:
    from pypdf import PdfReader
    r = PdfReader(str(path))
    parts = []
    for page in r.pages:
        try:
            t = page.extract_text() or ""
        except Exception as e:
            t = f"\n[extract error: {e}]\n"
        parts.append(t)
    return "\n".join(parts)


def extract_pdfplumber(path: Path) -> str:
    import pdfplumber
    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            try:
                t = page.extract_text() or ""
            except Exception as e:
                t = f"\n[extract error: {e}]\n"
            parts.append(t)
    return "\n".join(parts)


def extract(path: Path) -> str:
    t = extract_pdfplumber(path)
    if len(t.strip()) < 200:
        t2 = extract_pypdf(path)
        if len(t2.strip()) > len(t.strip()):
            t = t2
    return t


KEYWORDS = re.compile(
    r"(evaluation|metric|auc|roc|precision|recall|f1|f-score|fpr|false positive|"
    r"pr-auc|average precision|ap\b|session|user-level|day-level|event-level|point-wise|"
    r"insider threat|experiment|result|baseline|accuracy|confusion)",
    re.I,
)


def windows_around_matches(text: str, win: int = 1200, max_windows: int = 25) -> list[str]:
    text_lower = text.lower()
    spans = []
    for m in KEYWORDS.finditer(text):
        start = max(0, m.start() - win)
        end = min(len(text), m.end() + win)
        spans.append((start, end))
    if not spans:
        return [text[:8000]]
    merged = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1] + 50:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    chunks = [text[s:e] for s, e in merged[:max_windows]]
    return chunks


def main():
    pairs = resolve_paths()
    out_dir = Path(__file__).resolve().parent / "_pdf_extracts"
    out_dir.mkdir(exist_ok=True)
    for label, path in pairs:
        safe = re.sub(r"[^\w\-]+", "_", path.stem)[:80]
        print(f"\n{'='*70}\nFILE: {path.name}\nLABEL: {label}\n{'='*70}")
        try:
            full = extract(path)
        except Exception as e:
            print(f"EXTRACT FAIL: {e}")
            continue
        (out_dir / f"{safe}_full.txt").write_text(full, encoding="utf-8", errors="replace")
        chunks = windows_around_matches(full)
        snippet = "\n\n---SNIP---\n\n".join(chunks)
        (out_dir / f"{safe}_snips.txt").write_text(snippet, encoding="utf-8", errors="replace")
        print(f"chars full={len(full)} snips={len(snippet)}")
        # title guess: first 500 non-empty chars
        head = "\n".join(line.strip() for line in full.splitlines()[:40] if line.strip())[:1500]
        print("HEAD:\n", head[:800].encode(sys.stdout.encoding or 'utf-8', 'replace').decode(sys.stdout.encoding or 'utf-8'))


if __name__ == "__main__":
    main()
