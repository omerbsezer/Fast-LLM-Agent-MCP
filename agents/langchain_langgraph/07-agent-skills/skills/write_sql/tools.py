import re
from langchain_core.tools import tool

"""
Python tools for the write_sql skill.
Registered automatically when load_skill("write_sql") is called.
"""


@tool
def validate_sql_syntax(sql: str, dialect: str = "postgres") -> str:
    """Parse a SQL query and report syntax errors without executing it."""
    try:
        import sqlglot
        sqlglot.parse(sql, dialect=dialect, error_level=sqlglot.ErrorLevel.RAISE)
        return "✅ Valid SQL — no syntax errors detected."
    except ImportError:
        return "⚠️  sqlglot not installed. Run `pip install sqlglot`."
    except Exception as exc:
        return f"❌ Syntax error: {exc}"


@tool
def format_sql(sql: str, dialect: str = "postgres") -> str:
    """Pretty-print a SQL query using canonical formatting."""
    try:
        import sqlglot
        return sqlglot.transpile(sql, read=dialect, write=dialect, pretty=True)[0]
    except ImportError:
        return "⚠️  sqlglot not installed. Run `pip install sqlglot`."
    except Exception as exc:
        return f"❌ Could not format SQL: {exc}"


_RISKS = [
    (r"\bSELECT\s+\*\b",                   "🟡", "SELECT * — enumerate columns explicitly."),
    (r"\bIN\s*\(\s*SELECT\b",               "🟡", "IN (SELECT …) — prefer EXISTS or a JOIN."),
    (r"(?i)\bDELETE\s+FROM\b(?!.*\bWHERE\b)","🔴", "DELETE without WHERE — deletes ALL rows!"),
    (r"(?i)\bUPDATE\b(?!.*\bWHERE\b)",      "🔴", "UPDATE without WHERE — updates ALL rows!"),
    (r"(?i)\bDROP\s+(TABLE|DATABASE)\b",    "🔴", "DROP statement — destructive DDL."),
    (r"(?i)\bNOT\s+IN\s*\(\s*SELECT\b",    "🟡", "NOT IN (subquery) is NULL-unsafe — use NOT EXISTS."),
    (r"(?i)ORDER\s+BY\s+RAND\(\)",          "🟡", "ORDER BY RAND() is O(n log n) — slow on large tables."),
]


@tool
def detect_sql_risks(sql: str) -> str:
    """Scan a SQL query for common anti-patterns and pitfalls."""
    findings = [
        f"{lvl}: {msg}"
        for pattern, lvl, msg in _RISKS
        if re.search(pattern, sql, re.IGNORECASE)
    ]
    return "\n".join(findings) if findings else "✅ No obvious risks detected."