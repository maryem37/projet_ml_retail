# ==========================================
# RETAIL ML PROJECT - MAIN PIPELINE RUNNER
# ==========================================
# Usage:
#   python main.py                    → full pipeline
#   python main.py --no-flask         → without Flask
#   python main.py --mlflow           → with MLflow tracking
#   python main.py --test             → run pytest
#   python main.py --monitor          → run monitoring
#   python main.py --steps 1,3,4      → specific steps only
#   python main.py --skip-on-fail     → continue on failure
# ==========================================

import subprocess
import sys
import os
import time
import argparse
from datetime import datetime


STEPS = [
    { "id": 1, "name": "Preprocessing",         "script": "src/preprocessing.py",      "required": True  },
    { "id": 2, "name": "Clustering",             "script": "src/clustering.py",         "required": False },
    { "id": 3, "name": "Classification",         "script": "src/train_model.py",        "required": True  },
    { "id": 4, "name": "Tests (pytest)",         "script": "tests/",                    "required": False, "is_pytest": True  },
    { "id": 5, "name": "Monitoring",             "script": "src/monitoring.py",         "required": False },
    { "id": 6, "name": "Predict (Demo)",         "script": "src/predict.py",            "required": False },
    { "id": 7, "name": "Flask Web App",          "script": "app/app.py",                "required": False, "is_server": True  },
]


def separator(char="─", width=62):
    print(char * width)


def print_header():
    separator("═")
    print("  RETAIL ML PROJECT — PIPELINE")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    separator("═")


def print_step_banner(step, total):
    print()
    separator()
    print(f"  STEP {step['id']}/{total}  ·  {step['name'].upper()}")
    print(f"  Script : {step['script']}")
    separator()


def print_summary(results):
    print()
    separator("═")
    print("  PIPELINE SUMMARY")
    separator("═")
    icons = {"OK": "✅", "SKIP": "⏭ ", "FAIL": "❌"}
    for name, status, duration in results:
        icon = icons.get(status, "?")
        dur  = f"({duration:.1f}s)" if duration else ""
        print(f"  {icon}  {name:<38} {dur}")
    separator("═")
    failed = [n for n, s, _ in results if s == "FAIL"]
    if failed:
        print(f"\n  ❌ Failed steps: {', '.join(failed)}")
    else:
        print("\n  All steps completed successfully.")


def open_browser(url):
    import threading
    import webbrowser
    def _open():
        time.sleep(2)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def run_step(step):
    if not os.path.exists(step["script"]):
        print(f"  ⏭  Not found: {step['script']} — skipping.")
        return "SKIP", 0

    start = time.time()

    if step.get("is_pytest"):
        result = subprocess.run(
            [sys.executable, "-m", "pytest", step["script"], "-v", "--tb=short"],
            cwd=os.getcwd()
        )
    else:
        result = subprocess.run(
            [sys.executable, step["script"]],
            cwd=os.getcwd()
        )

    duration = time.time() - start
    status   = "OK" if result.returncode == 0 else "FAIL"
    icon     = "✅" if status == "OK" else "❌"
    print(f"\n  {icon} {step['name']} completed in {duration:.1f}s")
    return status, duration


def parse_args():
    p = argparse.ArgumentParser(description="Retail ML Pipeline")
    p.add_argument("--no-flask",     action="store_true", help="Skip Flask app")
    p.add_argument("--mlflow",       action="store_true", help="Use MLflow version of train_model")
    p.add_argument("--mlflow-ui",    action="store_true", help="Open MLflow UI after pipeline")
    p.add_argument("--test",         action="store_true", help="Run pytest tests")
    p.add_argument("--monitor",      action="store_true", help="Run monitoring")
    p.add_argument("--steps",        type=str, default=None, help="Run specific steps e.g. --steps 1,3")
    p.add_argument("--skip-on-fail", action="store_true", help="Continue pipeline on failure")
    return p.parse_args()


def main():
    args = parse_args()
    print_header()

    steps_to_run = STEPS.copy()

    # Switch to MLflow training script if requested and it exists
    if args.mlflow:
        mlflow_script = "src/train_model_mlflow.py"
        if os.path.exists(mlflow_script):
            steps_to_run = [
                {**s, "script": mlflow_script, "name": "Classification (MLflow)"}
                if s["id"] == 3 else s
                for s in steps_to_run
            ]
        else:
            print(f"  ⚠️  --mlflow requested but {mlflow_script} not found.")
            print(f"       Using src/train_model.py instead.")

    # Filter to specific step IDs if requested
    if args.steps:
        ids = [int(x.strip()) for x in args.steps.split(",")]
        steps_to_run = [s for s in steps_to_run if s["id"] in ids]

    # Remove Flask if --no-flask
    if args.no_flask:
        steps_to_run = [s for s in steps_to_run if not s.get("is_server")]

    # Only include pytest if --test flag given
    if not args.test:
        steps_to_run = [s for s in steps_to_run if not s.get("is_pytest")]

    # Only include monitoring if --monitor flag given
    if not args.monitor:
        steps_to_run = [s for s in steps_to_run if s["id"] != 5]

    total   = len(steps_to_run)
    results = []
    print(f"  Running {total} step(s)...\n")

    for step in steps_to_run:

        if step.get("is_server"):
            print_step_banner(step, total)
            print("  Launching Flask app → http://localhost:5000")

            if args.mlflow_ui:
                print("  Also launching MLflow UI → http://localhost:5001")
                subprocess.Popen(
                    [sys.executable, "-m", "mlflow", "ui", "--port", "5001"],
                    cwd=os.getcwd()
                )
                time.sleep(1)
                open_browser("http://localhost:5001")

            open_browser("http://localhost:5000")
            subprocess.run([sys.executable, step["script"]], cwd=os.getcwd())
            results.append((step["name"], "OK", None))
            break

        print_step_banner(step, total)
        status, duration = run_step(step)
        results.append((step["name"], status, duration))

        if status == "FAIL" and step.get("required") and not args.skip_on_fail:
            print(f"\n  ❌ Required step '{step['name']}' failed.")
            print(f"     Use --skip-on-fail to continue past failures.")
            print_summary(results)
            sys.exit(1)

    if args.mlflow_ui and not any(s.get("is_server") for s in steps_to_run):
        print("\n  Opening MLflow UI → http://localhost:5001")
        open_browser("http://localhost:5001")
        subprocess.run(
            [sys.executable, "-m", "mlflow", "ui", "--port", "5001"],
            cwd=os.getcwd()
        )

    print_summary(results)


if __name__ == "__main__":
    main()