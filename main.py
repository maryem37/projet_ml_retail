# ==========================================
# RETAIL ML PROJECT - MAIN PIPELINE RUNNER
# ==========================================
# Runs the full ML pipeline in the correct order:
#   1. Preprocessing
#   2. Utils (exploration)
#   3. Clustering
#   4. Classification (train_model)
#   5. Regression
#   6. Optuna tuning
#   7. Predict (demo)
#   8. Flask app (web interface)
#
# Usage:
#   python main.py             → full pipeline + Flask
#   python main.py --no-flask  → pipeline only, no Flask
#   python main.py --no-optuna → skip Optuna (slow)
#   python main.py --steps 1,3,4 → run only steps 1, 3 and 4
# ==========================================

import subprocess
import sys
import os
import time
import argparse
from datetime import datetime


# ==========================================
# CONFIGURATION
# ==========================================

STEPS = [
    {
        "id"     : 1,
        "name"   : "Preprocessing",
        "desc"   : "Clean, encode, engineer features, split data",
        "script" : "src/preprocessing.py",
        "required": True,
    },
    {
        "id"     : 2,
        "name"   : "Utils / Exploration",
        "desc"   : "Correlation heatmap, PCA, VIF analysis",
        "script" : "src/utils.py",
        "required": False,
    },
    {
        "id"     : 3,
        "name"   : "Clustering",
        "desc"   : "K-Means customer segmentation",
        "script" : "src/clustering.py",
        "required": False,
    },
    {
        "id"     : 4,
        "name"   : "Classification — Train Model",
        "desc"   : "Churn prediction with SMOTE + GridSearchCV",
        "script" : "src/train_model.py",
        "required": True,
    },
    {
        "id"     : 5,
        "name"   : "Regression",
        "desc"   : "Customer spend prediction (MonetaryTotal)",
        "script" : "src/regression.py",
        "required": False,
    },
    {
        "id"     : 6,
        "name"   : "Optuna Tuning",
        "desc"   : "Bayesian hyperparameter optimization",
        "script" : "src/tune_optuna.py",
        "required": False,
        "slow"   : True,   # flagged as slow — can be skipped
    },
    {
        "id"     : 7,
        "name"   : "Predict (Demo)",
        "desc"   : "Single customer prediction test",
        "script" : "src/predict.py",
        "required": False,
    },
    {
        "id"     : 8,
        "name"   : "Flask Web App",
        "desc"   : "Launch web interface on http://localhost:5000",
        "script" : "app/app.py",
        "required": False,
        "is_server": True,  # runs indefinitely — launched last
    },
]


# ==========================================
# HELPERS
# ==========================================

def separator(char="─", width=60):
    print(char * width)

def print_header():
    separator("═")
    print("  🛍️  RETAIL ML PROJECT — FULL PIPELINE RUNNER")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    separator("═")
    print()

def print_step_banner(step, index, total):
    print()
    separator()
    print(f"  STEP {step['id']}/{total}  ·  {step['name'].upper()}")
    print(f"  {step['desc']}")
    print(f"  Script  : {step['script']}")
    separator()

def print_summary(results):
    print()
    separator("═")
    print("  PIPELINE SUMMARY")
    separator("═")
    for name, status, duration in results:
        icon = "✅" if status == "OK" else ("⏭️ " if status == "SKIP" else "❌")
        dur  = f"({duration:.1f}s)" if duration else ""
        print(f"  {icon}  {name:<35} {dur}")
    separator("═")

def open_browser():
    """Opens browser to localhost:5000 after a short delay."""
    import threading
    import webbrowser

    def _open():
        time.sleep(2)  # wait for Flask to start
        print("\n  🌐 Opening browser → http://localhost:5000")
        webbrowser.open("http://localhost:5000")

    t = threading.Thread(target=_open, daemon=True)
    t.start()


# ==========================================
# RUN A SINGLE STEP
# ==========================================

def run_step(step):
    """
    Runs a single pipeline step as a subprocess.
    Returns (status, duration_seconds).
    """
    script = step["script"]

    if not os.path.exists(script):
        print(f"  ⚠️  Script not found: {script} — skipping.")
        return "SKIP", 0

    start = time.time()

    result = subprocess.run(
        [sys.executable, script],
        cwd=os.getcwd()
    )

    duration = time.time() - start

    if result.returncode == 0:
        print(f"\n  ✅ {step['name']} completed in {duration:.1f}s")
        return "OK", duration
    else:
        print(f"\n  ❌ {step['name']} FAILED (exit code {result.returncode})")
        return "FAIL", duration


# ==========================================
# ARGUMENT PARSER
# ==========================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Retail ML Pipeline Runner",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--no-flask",
        action="store_true",
        help="Run pipeline without launching Flask app"
    )

    parser.add_argument(
        "--no-optuna",
        action="store_true",
        help="Skip Optuna tuning step (can be slow)"
    )

    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Run only specific steps (e.g. --steps 1,3,4)"
    )

    parser.add_argument(
        "--skip-on-fail",
        action="store_true",
        help="Continue pipeline even if a step fails"
    )

    return parser.parse_args()


# ==========================================
# MAIN
# ==========================================

def main():
    args = parse_args()

    print_header()

    # Filter steps based on arguments
    steps_to_run = STEPS.copy()

    # Filter by --steps flag
    if args.steps:
        selected_ids = [int(s.strip()) for s in args.steps.split(",")]
        steps_to_run = [s for s in steps_to_run if s["id"] in selected_ids]
        print(f"  Running selected steps: {selected_ids}\n")

    # Remove Flask if --no-flask
    if args.no_flask:
        steps_to_run = [s for s in steps_to_run if not s.get("is_server")]
        print("  ℹ️  Flask app skipped (--no-flask)\n")

    # Remove Optuna if --no-optuna
    if args.no_optuna:
        steps_to_run = [s for s in steps_to_run if s["id"] != 6]
        print("  ℹ️  Optuna tuning skipped (--no-optuna)\n")

    total   = len(steps_to_run)
    results = []

    # Run each step
    for step in steps_to_run:

        # Flask server — launch last and open browser
        if step.get("is_server"):
            print_step_banner(step, total, total)
            print("  🌐 Launching Flask web application...")
            print("  → http://localhost:5000")
            print("  → Press CTRL+C to stop\n")
            open_browser()

            # Run Flask (blocking — stays running)
            subprocess.run([sys.executable, step["script"]], cwd=os.getcwd())
            results.append((step["name"], "OK", None))
            break

        # Regular step
        print_step_banner(step, step["id"], total)
        status, duration = run_step(step)
        results.append((step["name"], status, duration))

        # Stop pipeline on failure (unless --skip-on-fail)
        if status == "FAIL" and step.get("required") and not args.skip_on_fail:
            print(f"\n  ❌ Required step '{step['name']}' failed.")
            print("     Fix the error and re-run, or use --skip-on-fail to continue anyway.")
            print_summary(results)
            sys.exit(1)

    print_summary(results)


if __name__ == "__main__":
    main()