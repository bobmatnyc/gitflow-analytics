#!/usr/bin/env python3
"""Debug script to test .env loading and JIRA configuration."""

import os
import sys
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def debug_env_loading(config_path: str):
    """Debug environment variable loading and JIRA configuration."""

    print("\n" + "=" * 60)
    print("🔍 ENVIRONMENT VARIABLE LOADING DEBUG")
    print("=" * 60)

    config_path = Path(config_path)

    # 1. Check if config file exists
    print(f"\n📁 Config file: {config_path}")
    if not config_path.exists():
        print("   ❌ Config file does not exist!")
        return
    print("   ✅ Config file exists")

    # 2. Check for .env file
    config_dir = config_path.parent
    env_file = config_dir / ".env"
    print(f"\n📁 Looking for .env in: {config_dir}")
    print(f"   .env path: {env_file}")
    if env_file.exists():
        print("   ✅ .env file exists")

        # Show .env contents (without sensitive data)
        with open(env_file) as f:
            lines = f.readlines()
            print(f"   📋 .env file has {len(lines)} lines")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key = line.split("=")[0]
                        print(f"      - {key}=<value>")
    else:
        print("   ❌ .env file not found!")

    # 3. Check current environment variables (before loading)
    print("\n🔄 Environment variables BEFORE loading:")
    jira_user = os.environ.get("JIRA_ACCESS_USER")
    jira_token = os.environ.get("JIRA_ACCESS_TOKEN")
    github_token = os.environ.get("GITHUB_TOKEN")

    print(f"   JIRA_ACCESS_USER: {'✅ Set' if jira_user else '❌ Not set'}")
    print(f"   JIRA_ACCESS_TOKEN: {'✅ Set' if jira_token else '❌ Not set'}")
    print(f"   GITHUB_TOKEN: {'✅ Set' if github_token else '❌ Not set'}")

    # 4. Load configuration using ConfigLoader
    print("\n📋 Loading configuration with ConfigLoader...")
    try:
        from gitflow_analytics.config import ConfigLoader

        cfg = ConfigLoader.load(config_path)
        print("   ✅ Configuration loaded successfully")

        # 5. Check environment variables AFTER loading
        print("\n🔄 Environment variables AFTER loading:")
        jira_user = os.environ.get("JIRA_ACCESS_USER")
        jira_token = os.environ.get("JIRA_ACCESS_TOKEN")
        github_token = os.environ.get("GITHUB_TOKEN")

        print(f"   JIRA_ACCESS_USER: {'✅ Set' if jira_user else '❌ Not set'}")
        if jira_user:
            print(f"      Value: {jira_user[:3]}...{jira_user[-3:] if len(jira_user) > 6 else ''}")
        print(f"   JIRA_ACCESS_TOKEN: {'✅ Set' if jira_token else '❌ Not set'}")
        if jira_token:
            print(f"      Length: {len(jira_token)} chars")
        print(f"   GITHUB_TOKEN: {'✅ Set' if github_token else '❌ Not set'}")

        # 6. Check resolved JIRA configuration
        print("\n🔧 Resolved JIRA configuration:")
        if hasattr(cfg, "jira") and cfg.jira:
            print("   ✅ JIRA config exists")
            print(f"   access_user: {cfg.jira.access_user if cfg.jira.access_user else '❌ None'}")
            if cfg.jira.access_user:
                print(
                    f"      Value: {cfg.jira.access_user[:3]}...{cfg.jira.access_user[-3:] if len(cfg.jira.access_user) > 6 else ''}"
                )
            print(f"   access_token: {'✅ Set' if cfg.jira.access_token else '❌ None'}")
            if cfg.jira.access_token:
                print(f"      Length: {len(cfg.jira.access_token)} chars")
            print(f"   base_url: {cfg.jira.base_url if cfg.jira.base_url else '❌ None'}")
        else:
            print("   ❌ No JIRA configuration in loaded config")

        # 7. Check PM integration configuration
        print("\n🔧 PM Integration configuration:")
        if hasattr(cfg, "pm_integration") and cfg.pm_integration:
            print("   ✅ PM integration config exists")
            print(f"   enabled: {cfg.pm_integration.enabled}")
            if hasattr(cfg.pm_integration, "platforms"):
                print(
                    f"   platforms: {list(cfg.pm_integration.platforms.keys()) if cfg.pm_integration.platforms else 'None'}"
                )
                if "jira" in cfg.pm_integration.platforms:
                    jira_plat = cfg.pm_integration.platforms["jira"]
                    print("   JIRA platform:")
                    print(
                        f"      enabled: {jira_plat.enabled if hasattr(jira_plat, 'enabled') else 'N/A'}"
                    )
                    if hasattr(jira_plat, "config"):
                        print(
                            f"      config keys: {list(jira_plat.config.keys()) if jira_plat.config else 'None'}"
                        )
                        if jira_plat.config:
                            for key in ["username", "api_token", "base_url"]:
                                if key in jira_plat.config:
                                    val = jira_plat.config[key]
                                    if val:
                                        if key == "api_token":
                                            print(f"      {key}: ✅ Set ({len(val)} chars)")
                                        else:
                                            print(
                                                f"      {key}: {val[:10]}..."
                                                if len(str(val)) > 10
                                                else f"      {key}: {val}"
                                            )
                                    else:
                                        print(f"      {key}: ❌ None/Empty")
        else:
            print("   ❌ No PM integration configuration")

    except Exception as e:
        print(f"   ❌ Error loading configuration: {e}")
        import traceback

        traceback.print_exc()

    # 8. Test direct dotenv loading
    print("\n🔧 Testing direct dotenv loading:")
    try:
        from dotenv import load_dotenv

        # Clear JIRA vars first
        os.environ.pop("JIRA_ACCESS_USER", None)
        os.environ.pop("JIRA_ACCESS_TOKEN", None)

        print("   Cleared JIRA environment variables")
        print(f"   Loading .env from: {env_file}")
        result = load_dotenv(env_file, override=True)
        print(f"   load_dotenv result: {result}")

        jira_user = os.environ.get("JIRA_ACCESS_USER")
        jira_token = os.environ.get("JIRA_ACCESS_TOKEN")

        print(f"   JIRA_ACCESS_USER after dotenv: {'✅ Set' if jira_user else '❌ Not set'}")
        print(f"   JIRA_ACCESS_TOKEN after dotenv: {'✅ Set' if jira_token else '❌ Not set'}")

    except Exception as e:
        print(f"   ❌ Error testing dotenv: {e}")

    print("\n" + "=" * 60)
    print("🏁 DEBUG COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_env.py <config-file-path>")
        sys.exit(1)

    debug_env_loading(sys.argv[1])
