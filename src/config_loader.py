# ==========================================
# RETAIL ML PROJECT - CONFIG LOADER
# ==========================================
# Charge config.yaml et expose les parametres
# a tous les scripts du projet.
# ==========================================

import yaml
import os
import logging
from typing import Any, Dict, Optional

_config: Optional[Dict[str, Any]] = None

def _load_yaml_with_fallback(path: str) -> Dict[str, Any]:
    encodings = ("utf-8", "utf-8-sig", "utf-16", "latin-1")
    last_exc = None
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                data = yaml.safe_load(f)
                return data if isinstance(data, dict) else {}
        except UnicodeDecodeError as e:
            last_exc = e
            continue
    # Last resort: read binary and decode ignoring errors
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="ignore")
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        raise last_exc or e

def get_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Charge et met en cache la configuration depuis config.yaml (encodages fallback)."""
    global _config
    if _config is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"config.yaml not found at: {path}")
        _config = _load_yaml_with_fallback(path)
    return _config

def get_logger(name: str) -> logging.Logger:
    cfg = get_config()
    log_cfg = cfg.get("logging", {})

    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(getattr(logging, log_cfg.get("level", "INFO")))

        formatter = logging.Formatter(
            fmt=log_cfg.get("format", "%(asctime)s | %(levelname)s | %(message)s"),
            datefmt=log_cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")
        )

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_cfg.get("file", "logs/pipeline.log"), encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

if __name__ == "__main__":
    cfg = get_config()
    print("Config loaded successfully:")
    print(f"  Project : {cfg.get('project', {}).get('name')}")
    print(f"  Version : {cfg.get('project', {}).get('version')}")
    print(f"  MLflow  : {cfg.get('mlflow', {}).get('experiment_name')}")