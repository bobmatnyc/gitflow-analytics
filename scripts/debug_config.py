#!/usr/bin/env python3

from pathlib import Path

from gitflow_analytics.config import ConfigLoader

config_path = Path("ewtn-test/config-security-test.yaml")
loader = ConfigLoader()
cfg = loader.load(config_path)

print("Has security attr?", hasattr(cfg.analysis, "security"))
if hasattr(cfg.analysis, "security"):
    print("Security config:", cfg.analysis.security)
    print("Type:", type(cfg.analysis.security))
