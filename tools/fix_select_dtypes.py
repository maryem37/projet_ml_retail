import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# patterns to fix
PATTERNS = [
    (re.compile(r'select_dtypes\(\s*include\s*=\s*\[([^\]]*?["\']\s*str[^\]]*?)\]'), 
     "select_dtypes(include=[np.number])"),  # numeric selection (was trying to get str cols)
    (re.compile(r'select_dtypes\(\s*exclude\s*=\s*\[([^\]]*?)\]\s*\)'),
     "select_dtypes(include=[np.number])"),  # if already had exclude, convert to include numeric
]

def ensure_numpy_import(text: str) -> str:
    if "import numpy as np" in text:
        return text
    lines = text.splitlines()
    last_import = -1
    for i, ln in enumerate(lines):
        if ln.startswith("import ") or ln.startswith("from "):
            last_import = i
    insert_at = last_import + 1 if last_import >= 0 else 0
    lines.insert(insert_at, "import numpy as np")
    return "\n".join(lines)

def fix_file(p: Path, apply: bool):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if "select_dtypes" not in txt:
        return
    
    new = txt
    changed = False
    for pat, repl in PATTERNS:
        if pat.search(new):
            new = pat.sub(repl, new)
            changed = True
    
    if changed:
        print("PATCH:", p)
        bak = p.with_suffix(p.suffix + ".bak")
        if bak.exists():
            bak.unlink()
        shutil.copy2(p, bak)
        new = ensure_numpy_import(new)
        if apply:
            p.write_text(new, encoding="utf-8")
            print("  -> applied")
        else:
            print("  -> dry-run")

def main(apply=False):
    for p in (ROOT / "src").rglob("*.py"):
        fix_file(p, apply)
    for p in (ROOT / "app").rglob("*.py"):
        fix_file(p, apply)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    main(apply=args.apply)