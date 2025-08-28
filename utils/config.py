from typing import Union, List
import yaml

def _load_config(cfg: Union[str, dict]) -> dict:
    """Accept a dict or a YAML file path and return a dict."""
    if isinstance(cfg, dict):
        return cfg
    with open(cfg, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _get(d: dict, keys: List[str], default=None):
    """Nested get with default: keys like ['dataset','segment_frames']."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur