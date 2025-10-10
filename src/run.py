#!/usr/bin/env python3
"""run.py

Enhanced script that executes emulation.py with support for web-based configuration.

Features
--------
- Runs emulation.py with YAML configuration files
- Supports both CLI and web-based configuration
- Logs all execution details including commands
- Graceful cleanup on abort

Usage
-----
```bash
# Run single config
python run.py /path/to/config.yaml

# Run all configs in directory
python run.py /path/to/config_folder

# Force overwrite existing logs
python run.py /path/to/config.yaml --overwrite

# Specify custom emulation script
python run.py /path/to/config.yaml --emulation-script ./custom_emulation.py
```
"""
from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Iterator
import resources.clean_containernet as clean_containernet

###############################################################################
# User-replaceable hook
###############################################################################

def on_abort() -> None:
    """Cleanup function called on abort/interrupt"""
    clean_containernet.clean_session()

###############################################################################
# Logging helpers
###############################################################################

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"

def _setup_logger(log_file: Path) -> logging.Logger:
    """Create a logger that streams to stdout *and* a dedicated log file."""
    logger = logging.getLogger(log_file.stem)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(_LOG_FORMAT)

    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Stream handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Avoid duplicate logs if multiple handlers are added elsewhere
    logger.propagate = False

    return logger

###############################################################################
# Core functionality
###############################################################################

def _discover_configs(source: Path) -> Iterator[Path]:
    """Yield all YAML config files represented by *source* (file or directory)."""
    if source.is_file():
        if source.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Config file {source} must have .yaml/.yml extension")
        yield source.resolve()
    elif source.is_dir():
        for pattern in ("*.yaml", "*.yml"):
            for cfg in source.glob(pattern):
                if cfg.is_file():
                    yield cfg.resolve()
    else:
        raise FileNotFoundError(f"{source} does not exist or is not accessible")

def _default_logfile_for(config: Path) -> Path:
    """Return the path for the log file one directory *above* the config file."""
    parent_dir = config.parent.parent if config.parent.parent.exists() else config.parent
    log_dir = parent_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{config.stem}_{timestamp}.log"

def _existing_log_for(config: Path) -> Path | None:
    """Check if a log file already exists for the given config (same stem)."""
    log_dir = config.parent.parent / "logs" if (config.parent.parent).exists() else config.parent / "logs"
    if not log_dir.exists():
        return None
    for file in log_dir.glob(f"{config.stem}_*.log"):
        return file  # Return first match
    return None

def _log_execution_summary(config: Path, log_file: Path, returncode: int, 
                          start_time: datetime, end_time: datetime) -> None:
    """Log execution summary in JSON format"""
    summary = {
        "config_file": str(config),
        "log_file": str(log_file),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "return_code": returncode,
        "status": "success" if returncode == 0 else "failed"
    }
    
    summary_file = log_file.parent / f"{config.stem}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

def _run_single_config(config: Path, emulate_script: Path) -> None:
    """Spawn emulation.py with *config*, teeing output to console & log."""
    
    log_file = _default_logfile_for(config)
    logger = _setup_logger(log_file)

    workdir = emulate_script.parent
    cmd = [sys.executable, str(emulate_script), "--config", str(config)]
    
    logger.info("="*80)
    logger.info("STARTING EMULATION")
    logger.info("="*80)
    logger.info("Config: %s", config)
    logger.info("Working directory: %s", workdir)
    logger.info("Command: %s", " ".join(cmd))
    logger.info("="*80)
    
    start_time = datetime.now()

    try:
        with subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        ) as proc:
            assert proc.stdout
            for line in proc.stdout:
                logger.info(line.rstrip())
            
            returncode = proc.wait()
            end_time = datetime.now()
            
            if returncode:
                logger.error("="*80)
                logger.error("Process exited with non-zero status %s", returncode)
                logger.error("="*80)
            else:
                logger.info("="*80)
                logger.info("Process completed successfully")
                logger.info("="*80)
            
            # Log execution summary
            _log_execution_summary(config, log_file, returncode, start_time, end_time)
            
    except KeyboardInterrupt:
        logger.warning("="*80)
        logger.warning("Interrupted by user (Ctrl-C)")
        logger.warning("="*80)
        on_abort()
        sys.exit(130)
    except Exception:
        logger.exception("="*80)
        logger.exception("Unhandled exception during emulation run")
        logger.exception("="*80)
        on_abort()
        raise

###############################################################################
# Entry point
###############################################################################

def _install_signal_handlers() -> None:
    """Hook SIGINT/SIGTERM so cleanup fires even if argparse hasn't run yet."""
    def _handler(signum, _frame):
        print(f"\nSignal {signum} received — aborting…", file=sys.stderr)
        on_abort()
        sys.exit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)

def main() -> None:
    _install_signal_handlers()
    
    script_dir = Path(__file__).resolve().parent
    default_emulation = script_dir / "emulation.py"
    
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Run emulation.py for a YAML config or every YAML inside a directory.",
        epilog="Enhanced with web UI support and comprehensive logging."
    )
    parser.add_argument(
        "config_source",
        help="Path to YAML file or directory containing YAML files.",
    )
    parser.add_argument(
        "--emulation-script",
        default=None,
        help="Path to emulation.py (default: ./emulation.py)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-run even if a log already exists for a config file.",
    )
    parser.add_argument(
        "--web-ui",
        action="store_true",
        help="Launch web UI for configuration (ignores config_source)",
    )
    args = parser.parse_args()

    # Launch web UI if requested
    if args.web_ui:
        print("Starting web UI server...")
        web_server_path = script_dir / "web_server.py"
        if not web_server_path.exists():
            sys.exit(f"Web server script not found: {web_server_path}")
        
        subprocess.run([sys.executable, str(web_server_path)])
        return

    emulate_script = (
        Path(args.emulation_script).expanduser().resolve()
        if args.emulation_script
        else default_emulation
    )
    if not emulate_script.is_file():
        sys.exit(f"Emulation script not found: {emulate_script}")

    source = Path(args.config_source).expanduser().resolve()

    configs = list(_discover_configs(source))
    if not configs:
        sys.exit("No YAML configuration files found.")

    print(f"\nFound {len(configs)} configuration file(s) to process\n")

    for idx, cfg in enumerate(configs, 1):
        existing_log = _existing_log_for(cfg)
        if existing_log and not args.overwrite:
            print(f"[{idx}/{len(configs)}] Skipping {cfg.name} — log already exists at {existing_log}")
            continue
        
        print(f"\n[{idx}/{len(configs)}] Processing: {cfg.name}")
        print("-" * 80)
        
        clean_containernet.clean_session()
        _run_single_config(cfg, emulate_script)
        
        print("-" * 80)
        print(f"[{idx}/{len(configs)}] Completed: {cfg.name}\n")

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*80)

if __name__ == "__main__":
    main()