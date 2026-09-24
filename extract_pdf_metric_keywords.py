"""
Extract lines matching metric/granularity keywords from PDFs on Desktop,
or report missing files. Requires: pip install pymupdf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Install: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

DESKTOP = Path(r"c:\Users\Zhang\Desktop")
TARGETS = [
    ("Log2vec (CCS)", "3319535.3363224.pdf"),
    ("ITDBERT", "ITDBERT_ Temporal-semantic Representation for Insider Threat Detection.pdf"),
    ("HNTPP / Insider TPP", "Insider_Threat_Detection_via_Hierarchical_Neural_Temporal_Point_Processes.pdf"),
    ("LaAeb", "LaAeb _ A comprehensive log-text analysis based approach for insider threat detection - 1-s2.0-S0167404824004310-main.pdf"),
    ("DD-GCN", "A_High_Accuracy_and_Adaptive_Anomaly_Detection_Model_With_Dual-Domain_Graph_Convolutional_Network_for_Insider_Threat_Detection.pdf"),
    ("Tuor et al.", "1710.00811v2.pdf"),
]

KEYWORDS = re.compile(
    r"(?i)\b(metric|evaluation|ROC|AUC|precision|recall|F1|FPR|top[- ]?\d*|@\d+|budget|"
    r"user-level|day-level|session-level|event-level|log[- ]?entry|behavior-level|malicious\s+session)\b"
)


def extract_pdf_text(path: Path) -> str:
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    doc.close()
    return "\n".join(parts)


def main() -> None:
    for label, name in TARGETS:
        p = DESKTOP / name
        print("=" * 80)
        print(f"{label}: {p}")
        if not p.is_file():
            print("  [MISSING on Desktop]")
            continue
        text = extract_pdf_text(p)
        lines = text.splitlines()
        hits = []
        for i, line in enumerate(lines):
            if KEYWORDS.search(line):
                hits.append((i + 1, line.strip()[:240]))
        print(f"  pages ~ chars: {len(text)} | matching lines: {len(hits)}")
        for ln, snippet in hits[:120]:
            print(f"  L{ln}: {snippet}")
        if len(hits) > 120:
            print(f"  ... ({len(hits) - 120} more lines)")


if __name__ == "__main__":
    main()
