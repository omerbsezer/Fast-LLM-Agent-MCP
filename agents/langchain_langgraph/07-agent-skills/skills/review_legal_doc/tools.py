# review_legal_doc.py: registered automatically when load_skill("review_legal_doc") is called.
import re
from langchain_core.tools import tool

_CLAUSES = {
    "Liability Cap":                 [r"(?i)total\s+liability\s+shall\s+not\s+exceed", r"(?i)aggregate\s+liability.{0,40}limited\s+to"],
    "Consequential Damages Waiver":  [r"(?i)not\s+be\s+liable\s+for\s+any\s+indirect", r"(?i)indirect.{0,20}incidental.{0,20}consequential"],
    "Auto-Renewal":                  [r"(?i)auto.?renew", r"(?i)renews?\s+annually"],
    "Cancellation Notice":           [r"(?i)\d+\s+days?\s+(?:written\s+)?notice"],
    "IP Assignment":                 [r"(?i)intellectual\s+property.{0,40}assign", r"(?i)work\s+made\s+for\s+hire"],
    "Confidentiality":               [r"(?i)confidential\s+information", r"(?i)non.disclosure"],
    "Indemnification":               [r"(?i)indemnif(?:y|ication)", r"(?i)hold\s+harmless"],
    "Governing Law":                 [r"(?i)governed\s+by\s+the\s+laws?\s+of"],
    "Force Majeure":                 [r"(?i)force\s+majeure", r"(?i)acts?\s+of\s+God"],
    "Termination for Cause":         [r"(?i)terminat.{0,40}material\s+breach"],
}

_HIGH_RISK = {"Liability Cap", "Consequential Damages Waiver", "Auto-Renewal", "IP Assignment", "Indemnification"}
_RISK_WEIGHTS = {"Consequential Damages Waiver": 25, "Liability Cap": 20, "IP Assignment": 20,
                 "Indemnification": 15, "Auto-Renewal": 15, "Cancellation Notice": 10,
                 "Force Majeure": -5, "Confidentiality": -5, "Termination for Cause": -10}

def _found_clauses(text: str) -> set[str]:
    return {ct for ct, patterns in _CLAUSES.items() if any(re.search(p, text) for p in patterns)}

@tool
def extract_legal_clauses(text: str) -> str:
    """Scan contract text and return all detected clause types with context snippets."""
    results = []
    for clause_type, patterns in _CLAUSES.items():
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                start, end = max(0, m.start() - 30), min(len(text), m.end() + 90)
                snippet = text[start:end].replace("\n", " ").strip()
                icon = "🔴" if clause_type in _HIGH_RISK else "🟡"
                results.append(f"{icon} {clause_type}: …{snippet}…")
                break
    return "\n".join(results) if results else "No recognised clause types detected."

@tool
def score_legal_risk(text: str) -> str:
    """Return a heuristic risk score (0–100) with a LOW / MEDIUM / HIGH rating."""
    found = _found_clauses(text)
    score = max(0, min(100, sum(w for c, w in _RISK_WEIGHTS.items() if c in found)))
    rating = "🔴 HIGH" if score >= 60 else ("🟡 MEDIUM" if score >= 30 else "🟢 LOW")
    breakdown = "\n".join(
        f"  {'🔴' if w>0 else '🟢'} {c}: {'+' if w>0 else ''}{w} pts"
        for c, w in _RISK_WEIGHTS.items() if c in found
    )
    return f"Risk Score: {score}/100 → {rating}\n\nBreakdown:\n{breakdown}"

@tool
def extract_dates_and_deadlines(text: str) -> str:
    """Pull all date references and notice periods from contract text."""
    patterns = [r"\b\d+[\s-]?days?\b", r"\b\d+[\s-]?months?\b", r"\b\d+[\s-]?years?\b",
                r"\bann(?:ual(?:ly)?|um)\b", r"\b\d{4}-\d{2}-\d{2}\b"]
    seen, findings = set(), []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            key = m.group(0).lower()
            if key not in seen:
                seen.add(key)
                start, end = max(0, m.start()-40), min(len(text), m.end()+60)
                findings.append(f'  • "{m.group(0)}" — …{text[start:end].strip()}…')
    return "Dates & Deadlines:\n" + "\n".join(findings) if findings else "No date references found."