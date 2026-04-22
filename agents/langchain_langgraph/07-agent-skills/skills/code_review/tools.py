# code_review.md: registered automatically when load_skill("code_review") is called.
import ast
import re
import textwrap
from langchain_core.tools import tool

@tool
def detect_secrets(code: str) -> str:
    """Scan source code for hardcoded credentials, API keys, and connection strings."""
    patterns = [
        ("Hardcoded password",       r'(?i)password\s*=\s*["\'][^"\']{3,}["\']'),
        ("AWS Access Key",           r'AKIA[0-9A-Z]{16}'),
        ("Generic API key",          r'(?i)api[_-]?key\s*=\s*["\'][^"\']{8,}["\']'),
        ("Connection string w/creds",r'(?i)(?:postgres|mysql|mongodb)://[^:]+:[^@]+@'),
        ("Private key block",        r'-----BEGIN (?:RSA )?PRIVATE KEY-----'),
    ]
    findings = []
    for lineno, line in enumerate(code.splitlines(), 1):
        for label, pat in patterns:
            if re.search(pat, line):
                findings.append(f"  🔴 Line {lineno} — {label}: {re.sub(pat, '[REDACTED]', line).strip()}")
    return "Secret Scan:\n" + "\n".join(findings) if findings else "✅ No hardcoded secrets detected."


@tool
def analyze_python_ast(code: str) -> str:
    """Static analysis via Python AST: bare excepts, eval/exec, mutable defaults, long functions."""
    code = textwrap.dedent(code)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"❌ Syntax error: {exc}"

    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append(f"  🟡 Line {node.lineno}: Bare `except:` — use `except Exception:`.")
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name in ("eval", "exec"):
                issues.append(f"  🔴 Line {node.lineno}: `{name}()` — arbitrary code execution risk.")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in node.args.defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    issues.append(f"  🟡 Line {node.lineno}: `{node.name}` has mutable default argument.")
            length = node.end_lineno - node.lineno + 1
            if length > 50:
                issues.append(f"  🟡 Line {node.lineno}: `{node.name}` is {length} lines — consider splitting.")
        if isinstance(node, ast.Global):
            issues.append(f"  🟡 Line {node.lineno}: `global` statement — prefer explicit state passing.")

    return "AST Analysis:\n" + "\n".join(issues) if issues else "✅ No structural issues found."


@tool
def check_sql_injection(code: str) -> str:
    """Detect SQL injection via unsafe string interpolation in execute() calls."""
    patterns = [
        ("f-string in execute()",    r'\.execute\s*\(\s*f["\']'),
        ("% formatting in execute()",r'\.execute\s*\(\s*["\'][^"\']*%[^"\']*["\'\s]*%'),
        (".format() in execute()",   r'\.execute\s*\(\s*["\'][^"\']*\{.*?\}.*?\.format'),
        ("String concat in execute()",r'\.execute\s*\(\s*["\'][^"\']*["\'\s]*\+'),
    ]
    findings = []
    for lineno, line in enumerate(code.splitlines(), 1):
        for label, pat in patterns:
            if re.search(pat, line):
                findings.append(f"  🔴 Line {lineno} — {label}\n     Fix: use parameterised queries → execute(query, (value,))")
    return "SQL Injection Scan:\n" + "\n".join(findings) if findings else "✅ No SQL injection patterns detected."


@tool
def measure_complexity(code: str) -> str:
    """Estimate cyclomatic complexity per function (1–5 ✅, 6–10 🟡, 11–20 🟠, >20 🔴)."""
    code = textwrap.dedent(code)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"❌ Syntax error: {exc}"

    _BRANCH = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With, ast.Assert, ast.BoolOp)
    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            score = 1 + sum(1 for n in ast.walk(node) if isinstance(n, _BRANCH))
            icon = "🔴" if score > 20 else ("🟠" if score > 10 else ("🟡" if score > 5 else "✅"))
            results.append(f"  {icon} {node.name} (line {node.lineno}): {score}")

    return "Complexity Report:\n" + "\n".join(sorted(results)) if results else "No functions found."